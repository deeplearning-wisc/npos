from CLIP.CLIP_ft import CLIP_ft
import torch
import torchvision.datasets as dset
import os
import torch
import torchvision.transforms as trn
import torchvision.datasets.imagefolder as imagefolder
test_transform = trn.Compose([
    trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
    trn.CenterCrop(size=(224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
path = '/nobackup-slow/taoleitian/CLIP_feature/'
if not os.path.exists(path):
    os.makedirs(path)
model = CLIP_ft(num_classes=100, layers=8)
net = model
net = torch.nn.DataParallel(model, device_ids=list(range(1)))
root_dir = '/nobackup-slow/dataset/ILSVRC-2012/'
train_dir = root_dir + 'train'
train_data_in = imagefolder(train_dir, transform=test_transform)

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
