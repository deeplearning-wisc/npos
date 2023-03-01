import torchvision
import numpy as np
import sys
import pdb
import logging
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import seaborn as sns
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm
from torchvision import datasets,transforms
from .svhn_loader import SVHN

def set_loader(args):
    if args.in_dataset == 'CIFAR-10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if args.in_dataset == 'CIFAR-10':
        train_dataset = datasets.CIFAR10(root='./datasets/',
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root='./datasets/',
                                       train=False,
                                       transform=val_transform)
    elif args.in_dataset == 'CIFAR-100':
        train_dataset = datasets.CIFAR100(root='./datasets/',
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root='./datasets/',
                                        train=False,
                                        transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader

def set_ood_loader(args, out_dataset):
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    if args.in_dataset == 'CIFAR-10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if out_dataset == 'SVHN':
        testsetout = SVHN('/nobackup-slow/dataset/svhn/', split='test',
                                transform=transforms.Compose([transforms.ToTensor(), normalize]), download=False)
    elif out_dataset == 'cifar100':
        testsetout = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),normalize]))
    elif out_dataset == 'cifar10':
        testsetout = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),normalize]))
    elif out_dataset == 'LSUN_C':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/LSUN_C",
                                    transform=transforms.Compose([transforms.ToTensor(),normalize]))
    elif out_dataset == 'Places':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/places365",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
    elif out_dataset == 'iSUN':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/iSUN",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor(),normalize]))
    elif out_dataset == 'Textures':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
    else:
        testsetout = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/{}".format(out_dataset),
                                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
    if len(testsetout) > 10000: 
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=8)
    return testloaderOut

def obtain_feature_from_loader(net, loader, layer_idx, embedding_dim, num_batches):
    out_features = torch.zeros((0, embedding_dim), device = 'cuda')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if num_batches is not None:
                if batch_idx >= num_batches:
                    break
            data, target = data.cuda(), target.cuda()
            out_feature = net.intermediate_forward(data, layer_idx) 
            if layer_idx == 0: # out_feature: bz, 512, 4, 4
                out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1) #bz, 512, 16
                out_feature = torch.mean(out_feature, 2) # bz, 512
                out_feature =F.normalize(out_feature, dim = 1)
            out_features = torch.cat((out_features,out_feature), dim = 0)
    return out_features

def obtain_feature_from_scood_loader(net, loader, layer_idx, embedding_dim, num_batches):
    out_features = torch.zeros((0, embedding_dim), device = 'cuda')
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            data = sample['data']
            target = sample['label']
            if num_batches is not None:
                if batch_idx >= num_batches:
                    break
            data, target = data.cuda(), target.cuda()
            out_feature = net.intermediate_forward(data, layer_idx) 
            if layer_idx == 0: # out_feature: bz, 512, 4, 4
                out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1) #bz, 512, 16
                out_feature = torch.mean(out_feature, 2) # bz, 512
                out_feature =F.normalize(out_feature, dim = 1)
            out_features = torch.cat((out_features,out_feature), dim = 0)
    return out_features

def save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list):
    fpr_list = [float('{:.2f}'.format(100*fpr)) for fpr in fpr_list]
    auroc_list = [float('{:.2f}'.format(100*auroc)) for auroc in auroc_list]
    aupr_list = [float('{:.2f}'.format(100*aupr)) for aupr in aupr_list]
    import pandas as pd
    data = {k:v for k,v in zip(out_datasets, zip(fpr_list,auroc_list,aupr_list))}
    data['AVG'] = [np.mean(fpr_list),np.mean(auroc_list),np.mean(aupr_list) ]
    data['AVG']  = [float('{:.2f}'.format(metric)) for metric in data['AVG']]
    # Specify orient='index' to create the DataFrame using dictionary keys as rows
    df = pd.DataFrame.from_dict(data, orient='index', columns=['FPR95', 'AUROC', 'AUPR'])
    df.to_csv(os.path.join(args.log_directory,f'{args.name}.csv'))

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    # args.score = 'CLS'
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID": id_scores, "OOD":  ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    # plt.title(f"ID v.s. {out_dataset} {args.score} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join(args.log_directory,f"KNN_{out_dataset}.png"), bbox_inches='tight')