import torchvision
from torchvision.transforms import transforms
import numpy as np
import sys
import pdb
import logging
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.resnet_outliers import *
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import seaborn as sns
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm
from evaluation.eval_utils import *
from evaluation.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import evaluation.svhn_loader as svhn

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--name', default = "pretrained")
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dim')
    parser.add_argument('--model', default='resnet34', type=str, help='model architecture')
    parser.add_argument('--epoch',default=500,type=int)
    parser.add_argument('--K',default=100,type=int)
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,help='List of GPU indices to use, e.g., --gpus 0 1 2 3')
    parser.add_argument('--method_name', '-test', type=str, default='test', help='Method name.')
    parser.add_argument('--ckpt',  type=str, default=
    '/nobackup-slow/taoleitian/CIDER/paper_results/CIFAR-100_ckpt_500.pt',
    #'/nobackup-slow/taoleitian/CIDER/10_02_16:25_SupCon_resnet34_lr_0.5_cosine_True_supcon_ws_1_500_128_trial_0_linear_temp_0.1_CIFAR-100/checkpoint_500.pth.tar',
                        help='Method name.')

    args = parser.parse_args()

    # use 512
    args.name = '29_06_21:14_SupCon_resnet34_lr_0.05_warm_True_cosine_True_bsz_512_ws_1.0_wu_1.0_128_temp_0.1_CIFAR-100_pm_0.95'
    # args.name = '24_06_23:40_SupCon_resnet34_lr_0.5_cosine_True_bsz_512_triple_ws_0_wu_0.5_wp_1_500_128_trial_0_linear_temp_0.1_CIFAR-100_pm_0.95_momentum_norm'
    #args.ckpt = f"./checkpoints_save/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"
    #args.ckpt = f"./checkpoints_save/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"
    args.gpus = list(map(lambda x: torch.device('cuda', x), args.gpus))
    if args.in_dataset == "CIFAR-10":
        args.num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        args.num_classes = 100
    return args

def set_model(args):
    model = SupCEHeadResNet(name=args.model, feat_dim=args.feat_dim, num_classes=args.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    return model

def knn(layer_idx=0, num_classes=100):
    args = process_args()
    args.log_directory = f"results/{args.in_dataset}/{args.name}/knn_{args.K}"
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
   
    # setup model
    train_loader, test_loader = set_loader(args)
    print(args.ckpt)
    pretrained_dict= torch.load(args.ckpt,  map_location='cpu')
    print("Keys:", pretrained_dict.keys())
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    net = set_model(args)
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    if layer_idx == 1:
        embedding_dim = 128
    elif layer_idx == 0:
        embedding_dim = 512

    # extract features
    ftrain = obtain_feature_from_loader(net, train_loader, layer_idx, embedding_dim, num_batches=None)
    ftest = obtain_feature_from_loader(net, test_loader, layer_idx, embedding_dim, num_batches=None)
    print('ID finished')
    out_datasets = ['LSUN_C', 'iSUN', 'SVHN', 'Places', 'Textures']
    #out_datasets = ['LSUN_C', 'iSUN', 'SVHN', 'places365']
    # out_datasets = ['LSUN', 'isun', 'SVHN', 'places365', 'texture', 'Imagenet', 'tin']
    food_all = {}
    ood_num_examples = len(test_loader.dataset)
    num_batches = ood_num_examples // args.batch_size
    for out_dataset in out_datasets:
        ood_loader = set_ood_loader(args, out_dataset)
        ood_feat = obtain_feature_from_loader(net, ood_loader, layer_idx, embedding_dim, num_batches)
        food_all[out_dataset] = ood_feat
        print(f'OOD {out_dataset} finished')

    # initialization
    auroc_list, aupr_list, fpr_list = [], [], []
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain.cpu().numpy())
    index_bad = index
    ################### Using KNN distance Directly ###################
    D, _ = index_bad.search(ftest.cpu().numpy(), args.K,)
    scores_in = -D[:,-1]
    for ood_dataset, food in food_all.items():
        print(f"Evaluting OOD dataset {ood_dataset}")
        D, _ = index_bad.search(food.cpu().numpy(),args.K)
        scores_ood_test = -D[:,-1]
        aurocs, auprs, fprs = [], [], []
        print(scores_in, scores_ood_test)
        print(scores_in[:3], scores_ood_test[:3])
        torch.save(net.state_dict(), '/nobackup-fast/taoleitian/CIFAR_100.pt')
        measures = get_measures(scores_in, scores_ood_test)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
        print_measures(None, auroc, aupr, fpr, args.method_name)
        #plot_distribution(args, scores_in, scores_ood_test, ood_dataset)
    print("AVG")
    print_measures(None, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    knn(layer_idx=0, num_classes=10)
