import logging
import math
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_transforms(
    mean, std
):  
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    return TwoCropTransform(train_transform)

def get_test_transforms(
    mean, std
):  
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    return test_transform

def get_cifar10(args, root):
    transform = get_transforms(cifar10_mean, cifar10_std)
    test_transform = get_test_transforms(cifar10_mean, cifar10_std)

    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform)
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=transform)
    test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cifar100(args, root):
    transform = get_transforms(cifar100_mean, cifar100_std)
    test_transform = get_test_transforms(cifar100_mean, cifar100_std)

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform)
    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=transform)
    test_dataset = datasets.CIFAR100(root, train=False, download=True, transform=test_transform)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels, expand_labels=True):
    label_ratio = args.label_ratio
    num_labeled = int(label_ratio*len(labels))
    num_unlabeled = len(labels)-num_labeled
    print("Distribution:")
    print(num_labeled, num_unlabeled, len(labels))

    label_per_class = num_labeled // args.n_cls
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.n_cls):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        l_idx = idx[:label_per_class]
        u_idx = idx[label_per_class:]
        labeled_idx.extend(l_idx)
        unlabeled_idx.extend(u_idx)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == num_labeled
    assert len(unlabeled_idx) == num_unlabeled
    # # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # unlabeled_idx = np.array(range(len(labels)))

    if expand_labels or num_labeled < batch_size:
        num_iter = int(len(unlabeled_idx)/(args.mu*args.batch_size))
        num_expand_x = math.ceil(args.batch_size * num_iter / num_labeled)
        print("Expand:", num_expand_x)
        if num_expand_x!=0:
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None and len(indexs)>0:
            self.shrink_data(indexs)
            print(len(self.data), len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = torch.from_numpy(targets[idxs])
        self.data = self.data[idxs, ...]

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None and len(indexs)>0:
            self.shrink_data(indexs)
            print(len(self.data), len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = torch.from_numpy(targets[idxs])
        self.data = self.data[idxs, ...]

CIFAR_GETTERS = {'CIFAR-10': get_cifar10,
                   'CIFAR-100': get_cifar100}