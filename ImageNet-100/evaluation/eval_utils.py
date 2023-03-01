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
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import torchvision.transforms as trn
import torchvision.datasets as dset
import faiss
#from image_folder import ImageSubfolder
from torchvision import datasets,transforms
from .svhn_loader import SVHN
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


class ImageSubfolder(DatasetFolder):
    """Extend ImageFolder to support fold subsets
    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        class_to_idx (dict): Dict with items (class_name, class_index).
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_to_idx: Optional[Dict] = None,
    ):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        if class_to_idx is not None:
            classes = class_to_idx.keys()
        else:
            classes, class_to_idx = self.find_classes(self.root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None,
        samples = self.make_dataset(self.root, class_to_idx, extensions[0], is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
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
    elif args.in_dataset == 'ImageNet-100':
        test_transform = trn.Compose([
            trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
            trn.CenterCrop(size=(224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        root_dir = '/nobackup-slow/dataset/ILSVRC-2012/'
        print('Loading from' + root_dir)
        train_dir = root_dir + 'val'
        classes, _ = dset.folder.find_classes(train_dir)
        index = [125, 788, 630, 535, 474, 694, 146, 914, 447, 208, 182, 621, 271, 646, 328, 119, 772, 928, 610, 891,
                 340,
                 890, 589, 524, 172, 453, 869, 556, 168, 982, 942, 874, 787, 320, 457, 127, 814, 358, 604, 634, 898,
                 388,
                 618, 306, 150, 508, 702, 323, 822, 63, 445, 927, 266, 298, 255, 44, 207, 151, 666, 868, 992, 843, 436,
                 131,
                 384, 908, 278, 169, 294, 428, 60, 472, 778, 304, 76, 289, 199, 152, 584, 510, 825, 236, 395, 762, 917,
                 573,
                 949, 696, 977, 401, 583, 10, 562, 738, 416, 637, 973, 359, 52, 708]

        num_classes = 100
        classes = [classes[i] for i in index]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        train_dataset = ImageSubfolder(root_dir + 'train', transform=test_transform, class_to_idx=class_to_idx)
        val_dataset = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader

def set_ood_loader(args, out_dataset):
    test_transform = trn.Compose([
        trn.Resize(size=(224,224), interpolation=trn.InterpolationMode.BICUBIC),
        # trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        trn.RandomHorizontalFlip(p=0.5),
        trn.ToTensor(),
        trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
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
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/SUN",
                                    #transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
                                    transform=test_transform)
    elif out_dataset == 'Places':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Places",
                                    #transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
                                    transform=test_transform)

    elif out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/iNaturalist",
                                    #transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
                                    transform=test_transform)
    elif out_dataset == 'Texture':
        testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/dtd/images",
                                    transform=test_transform)
    else:
        testsetout = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/{}".format(out_dataset),
                                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),normalize]))
    if len(testsetout) > 10000: 
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=8)
    return testloaderOut

def obtain_feature_from_loader(net, loader, layer_idx, embedding_dim, num_batches):
    out_features = torch.zeros((0, embedding_dim), device = 'cpu')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx % 10==0:
                print(batch_idx)
            if num_batches is not None:
                if batch_idx >= num_batches:
                    break
            data, target = data.cuda(), target.cuda()
            #out_feature = net.intermediate_forward(data, layer_idx)
            #out_feature = net.intermediate_forward(data)
            out_feature = net(data, fc=False).cpu()
            if layer_idx == 0: # out_feature: bz, 512, 4, 4
                out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1) #bz, 512, 16
                out_feature = torch.mean(out_feature, 2) # bz, 512
                out_feature =F.normalize(out_feature, dim = 1).cpu()
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
            out_feature = net(data, fc=False).cpu()
            if layer_idx == 0: # out_feature: bz, 512, 4, 4
                out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1) #bz, 512, 16
                out_feature = torch.mean(out_feature, 2) # bz, 512
                out_feature = F.normalize(out_feature, dim = 1)
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