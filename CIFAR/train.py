import argparse
import math
from losses import SupConLoss
from models.resnet import SupCEResNet
import os
import time
from datetime import datetime
import logging
import tensorboard_logger as tb_logger
import pprint
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import models.densenet as dn
from losses import DispLoss, CompLoss
from util import adjust_learning_rate, warmup_learning_rate, accuracy, AverageMeter
from tensorboard_logger import configure, log_value
from cifar import CIFAR_GETTERS
from models.resnet import SupCEHeadResNet

parser = argparse.ArgumentParser(description='Training with Cross Entropy Loss')
parser.add_argument('--gpus', default=1, nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')
parser.add_argument('--w_disp', default= 0.5, type=float,
                    help='L uniform weight')
parser.add_argument('--w_comp', default= 1, type=float,
                    help='L uniform weight')
parser.add_argument('--proto_m', default= 0.95, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--model', default='resnet34', type=str)
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=1, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', default=0.00000000, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=30, type=int,
                    help='print frequency (default: 10)')
# learning settings
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cosine', action='store_false',
                        help='using cosine annealing')
parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
args = parser.parse_args()
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]



# dirs
date_time = datetime.now().strftime("%d_%m_%H:%M")
args.name = f"{date_time}_SupCon_{args.model}_lr_{args.learning_rate}_warm_{args.warm}_cosine_{args.cosine}_bsz_{args.batch_size}_disp_{args.w_disp}_comp_{args.w_comp}_{args.feat_dim}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}"
args.log_directory = "./logs/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
args.model_directory = "/nobackup-slow/taoleitian/CIDER/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
args.tb_path = './logs/{in_dataset}/{name}/'.format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, 'tb')
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)



#save args
state = {k: v for k, v in args._get_kwargs()}
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 
log.debug(state)

#setup GPU
#args.gpus = list(map(lambda x: torch.device('cuda', x), args.gpus))
if args.in_dataset == "CIFAR-10":
    args.n_cls = 10
elif args.in_dataset == "CIFAR-100":
    args.n_cls = 100

#set seeds
torch.manual_seed(20)
torch.cuda.manual_seed(20)
np.random.seed(20)
log.debug(f"{args.name}")

# warm-up for large-batch training,
if args.batch_size > 256:
    args.warm = True
if args.warm:
    args.warmup_from = 0.01
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate

args.mu = 7
args.label_ratio = 1.0
def set_loader(args):
    labeled_dataset, unlabeled_dataset, test_dataset = CIFAR_GETTERS[args.in_dataset](
        args, './datasets')
    labeled_trainloader = torch.utils.data.DataLoader(labeled_dataset, \
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_dataset, \
        batch_size=args.batch_size*args.mu, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_dataset,
         batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return labeled_trainloader, unlabeled_trainloader, testloader

def set_model(args):
    model = SupCEHeadResNet(name=args.model, feat_dim=args.feat_dim, num_classes=args.n_cls)
    criterion = nn.CrossEntropyLoss().cuda()

    # enable synchronized Batch Normalization
    if args.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
            #model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    # dataset
    train_loader, _, val_loader = set_loader(args)

    # training settings

    model, criterion_cls = set_model(args)
    criterion_disp = DispLoss(args, model, train_loader, temperature=args.temp).cuda()
    criterion_comp = CompLoss(args, temperature=args.temp).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint\

    if args.resume:
        #if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        #checkpoint = torch.load(args.resume)
        print(args.resume)
        #args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        #else:
            #print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        
        # train
        train_disp_loss, train_comp_loss, train_acc = train_epoch(args, train_loader, model, criterion_disp, criterion_comp, optimizer, epoch, log)
        tb_log.log_value('train_disp_loss', train_disp_loss, epoch)
        tb_log.log_value('train_comp_loss', train_comp_loss, epoch)
        tb_log.log_value('train_acc', train_acc, epoch)

        # evaluate
        val_acc = validate(val_loader, model, criterion_cls, epoch, log)
        tb_log.log_value('val_acc', val_acc, epoch)
        tb_log.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save checkpoint
        if (epoch + 1) % args.save_epoch == 0 and (epoch + 1) >= 0:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
            }, epoch + 1)

def train_epoch(args, train_loader, model, criterion_disp, criterion_comp, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    acc = AverageMeter()
    disp_losses = AverageMeter()
    comp_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target.shape[0]
        input = torch.cat([input[0], input[1]], dim=0).cuda()
        target = target.repeat(2).cuda()

        penultimate = model.encoder(input)
        features= model.head(penultimate)
        features= F.normalize(features, dim=1)
        disp_loss = criterion_disp(features, target)
        comp_loss = criterion_comp(features, criterion_disp.prototypes, target)
        pred = model.fc(penultimate)

        # measure accuracy and record loss
        acc_batch = accuracy(pred.data, target, topk=(1,))[0][0]
        acc.update(acc_batch, input.size(0))
        disp_losses.update(disp_loss.data, input.size(0))
        comp_losses.update(comp_loss.data, input.size(0))
        loss = args.w_disp* disp_loss + args.w_comp* comp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log.debug('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Disp Loss {uloss.val:.4f} ({uloss.avg:.4f})\t'
                'Comp Loss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,uloss=disp_losses,ploss=comp_losses,top1=acc))
    return disp_losses.avg, comp_losses.avg, acc.avg

def validate(val_loader, model, criterion, epoch, log):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc_batch = accuracy(output.data, target, topk=(1,))[0][0]
            losses.update(loss.data, input.size(0))
            acc.update(acc_batch, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=acc))
        log.debug('*Acc@ {top1.avg:.3f}'.format(top1=acc))
        return acc.avg

def save_checkpoint(args, state, epoch):
    """Saves checkpoint to disk"""
    filename = args.model_directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)

if __name__ == '__main__':
    main()