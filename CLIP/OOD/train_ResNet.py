# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from KNN.KNN import generate_outliers, generate_outliers_rand
from models.densenet import DenseNet3
from CLIP.CLIP_ft import clipnet_ft
from CLIP.CLIP_model import clipnet
from CLIP.CLIP_ResNet import CLIP_ResNet
from CLIP.clip_feature_dataset import clip_feature
import torchvision.transforms as trn
from CLIP.logitnorm_loss import LogitNormLoss
import faiss.contrib.torch_utils

from torch.distributions import MultivariateNormal


# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet-10','ImageNet-100'],
                    default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='dense',
                    choices=['allconv', 'wrn', 'dense'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=512, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# KNN-based OOD detection
parser.add_argument('--start_epoch', type=int, default=40, help='start epoch to use the outlier loss')
parser.add_argument('--num_layers', type=int, default=10, help='The number of layers to be fixed in CLIP')
parser.add_argument('--sample_number', type=int, default=1000, help='number of standard Gaussian noise samples')
parser.add_argument('--select', type=int, default=50, help='How many ID samples to pick to define as points near the boundary of the sample space')
parser.add_argument('--sample_from', type=int, default=1000, help='Number of IDs per class used to estimate OOD data.')
parser.add_argument('--T', type=int, default=10., help='temperature value')
parser.add_argument('--K', type=int, default=100, help='The value of top-K to calculate the KNN distance')
parser.add_argument('--loss_weight', type=float, default=0.1, help='The weight of outlier loss')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Learning rate decay ratio for MLP outlier')
parser.add_argument('--cov_mat', type=float, default=0.1, help='The weight before the covariance matrix to determine the sampling range')
parser.add_argument('--sampling_ratio', type=float, default=1., help='What proportion of points to choose to calculate the KNN value')
parser.add_argument('--ID_points_num', type=int, default=2, help='the number of synthetic outliers extracted for each selected ID')
parser.add_argument('--pick_nums', type=int, default=5, help='Number of ID samples used to generate outliers')



args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)


train_transform = trn.Compose([
    trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
    #trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
    trn.RandomHorizontalFlip(p=0.5),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
test_transform = trn.Compose([
    trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
    trn.CenterCrop(size=(224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
num_layers = args.num_layers
if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'ImageNet-100':
    load_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/ResNet/ImageNet-100'
    #load_path = '/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet-100/'
    train_data = clip_feature(path=load_path+'/train/')
    test_data  = clip_feature(path=load_path+'/val/')
    num_classes = 100
elif args.dataset == 'ImageNet-10':
    train_data = clip_feature(path='/nobackup-slow/dataset/ImageNet_OOD_dataset_feature/ImageNet-10/train/')
    test_data = clip_feature(path='/nobackup-slow/dataset/ImageNet_OOD_dataset_feature/ImageNet-10/val/')
    num_classes = 10
else:
    train_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 100


calib_indicator = ''
if args.calibration:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    calib_indicator = '_calib'

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

net = CLIP_ResNet(num_classes=num_classes, layers=num_layers)
net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/nobackup-slow/taoleitian/model/vos/ImageNet-100/MCM/ResNet/NPOS_2/ImageNet-100_dense_baseline_dense_epoch_9.pt').items()}, strict=False)






start_epoch = 0


cudnn.benchmark = True  # fire on all cylinders

if args.dataset == 'cifar10':
    num_classes = 10
else:
    num_classes = 100

number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
#logistic_regression = torch.nn.Linear(1, 2)
#logistic_regression = logistic_regression
optimizer = torch.optim.SGD([
    {"params": net.attnpool.parameters()},
    {"params": net.weight_energy.parameters(), "lr": state['learning_rate'] * args.decay_rate},
    ],
    lr=state['learning_rate'],
    momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60],
    gamma=10)
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))



# /////////////// Training ///////////////

def train(epoch, ood_list):
    net.train()  # enter train mode
    loss_avg = 0.0
    lr_reg_loss_avg = 0.0
    data_dict = torch.zeros(num_classes, args.sample_number, 1024).cuda()
    res = faiss.StandardGpuResources()
    KNN_index = faiss.GpuIndexFlatL2(res, 1024)
    criterion = LogitNormLoss()

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        x, output, energy_score_for_fg = net(data)

        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
            # Standard Gaussian distribution
            new_dis = MultivariateNormal(torch.zeros(1024).cuda(), torch.eye(1024).cuda())
            negative_samples = new_dis.rsample((args.sample_from,))
            for index in range(num_classes):
                ID = data_dict[index]
                #start_time = time.time()
                sample_point = generate_outliers(ID, input_index=KNN_index,
                                                 negative_samples=negative_samples, ID_points_num=2, K=args.K, select=args.select,
                                                 cov_mat=args.cov_mat, sampling_ratio=1.0, pic_nums=args.pick_nums, depth=1024)
                #end_time = time.time()
                #print("time cost:", float(end_time - start_time) * 1000.0, "ms")
                if index == 0:
                    ood_samples = sample_point
                    '''
                    if epoch >= 2:
                        ood_list.append(ood_samples.detach().cpu())
                    if epoch >= 2:
                        torch.save(data_dict[index].detach().cpu(), 'clip_ID_3.pt')
                    '''
                else:
                    ood_samples = torch.cat((ood_samples, sample_point), 0)

            if len(ood_samples) != 0:
                predictions_ood, energy_score_for_bg = net(ood_samples, fc=True)
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), 0).squeeze()
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)
                criterion_BCE = torch.nn.BCEWithLogitsLoss()
                lr_reg_loss = criterion_BCE(input_for_lr.view(-1), labels_for_lr)

        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward
        optimizer.zero_grad()
        loss = criterion(x, target)
        # breakpoint()
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()


        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        lr_reg_loss_avg = lr_reg_loss_avg * 0.8 + float(lr_reg_loss) * 0.2
    scheduler.step()
    state['train_loss'] = loss_avg
    state['lr_reg_loss'] = lr_reg_loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output, _, _ = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
'_' + str(args.loss_weight) + \
                             '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from) +
                                  '_dense_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')
ood_list = []
# Main loop
#data_dict = torch.zeros(num_classes, args.sample_number, 512).cuda()

for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch
    begin_epoch = time.time()

    train(epoch, ood_list)
    test()


    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                            '_baseline_dense_'  + 'epoch_'  + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                             '_baseline_dense_' + 'epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                      '_' + str(args.loss_weight) + \
                                      '_' + str(args.sample_number) + '_' + str(args.start_epoch) + '_' + \
                                      str(args.select) + '_' + str(args.sample_from) +
                                      '_dense_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['lr_reg_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Outliner Loss {5:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'],
        state['lr_reg_loss'])
    )
#ood_data = torch.cat(ood_list, dim=0)
#torch.save(ood_data, 'clip_outliers_3.pt')