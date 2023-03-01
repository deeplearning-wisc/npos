from clip_feature_dataset import clip_feature
from CLIP_model import CLIP_ft
import torch
import torchvision.datasets as dset
import os
import torch
import torchvision.transforms as trn
from image_folder import ImageSubfolder

test_transform = trn.Compose([
    trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
    trn.CenterCrop(size=(224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
path = '/nobackup-slow/taoleitian/imagenet-r_feature/'
if not os.path.exists(path):
    os.makedirs(path)
model = CLIP_ft(num_classes=100, layers=6)
net = model
net = torch.nn.DataParallel(model, device_ids=list(range(1)))
'''root_dir = '/nobackup-slow/dataset/ILSVRC-2012/'
train_dir = root_dir + 'val'
classes, _ = dset.folder.find_classes(train_dir)
index = [125, 788, 630, 535, 474, 694, 146, 914, 447, 208, 182, 621, 271, 646, 328, 119, 772, 928, 610, 891, 340,
             890, 589, 524, 172, 453, 869, 556, 168, 982, 942, 874, 787, 320, 457, 127, 814, 358, 604, 634, 898, 388,
             618, 306, 150, 508, 702, 323, 822, 63, 445, 927, 266, 298, 255, 44, 207, 151, 666, 868, 992, 843, 436, 131,
             384, 908, 278, 169, 294, 428, 60, 472, 778, 304, 76, 289, 199, 152, 584, 510, 825, 236, 395, 762, 917, 573,
             949, 696, 977, 401, 583, 10, 562, 738, 416, 637, 973, 359, 52, 708]

num_classes = 100
classes = [classes[i] for i in index]
class_to_idx = {c: i for i, c in enumerate(classes)}
train_data_in = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
'''
test_transform = trn.Compose([
    trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
    trn.CenterCrop(size=(224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
train_data_in = dset.ImageFolder(root="/nobackup-slow/taoleitian/imagenet-r/",
                            transform=test_transform)
train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=1024, shuffle=False,
    num_workers=8, pin_memory=True)
image_tensor_list = []
image_target_tensor_list = []
i = 0
for idx, data in enumerate(train_loader_in):
    print(idx)
    with torch.no_grad():
        embedding = net(input=data[0].cuda())
    image_tensor_list.append(embedding.half().cpu())
    image_target_tensor_list.append(data[1])

image_tensor = torch.cat(image_tensor_list, dim=0)
image_target_tensor = torch.cat(image_target_tensor_list, dim=0)
torch.save(image_tensor, path +"feature.pt")
torch.save(image_target_tensor, path +"target.pt")
