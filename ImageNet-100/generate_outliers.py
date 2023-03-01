import torch
import faiss
import time
from torch.distributions import MultivariateNormal

import faiss.contrib.torch_utils
from KNN import generate_outliers

dataset = torch.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet100/train/feature.pt')
label = torch.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet100/train/target.pt')
feature_list = []
for i in range(100):
    indexed_feature = []
    feature_list.append(indexed_feature)
for index in range(len(dataset)):
    feature = dataset[index:index+1]
    feature_list[label[index]].append(feature)
for i in range(100):
    feature_list[i] = torch.cat(feature_list[i], dim=0)
#torch.save(feature_list[0], '/nobackup-slow/taoleitian/CLIP_visual_feature/OOD/OOD/class_0.pt')
outliers_list = []
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 512)
new_dis = MultivariateNormal(torch.zeros(512),  torch.eye(512))
length = 30000
target = torch.zeros([200*1000, 1])
for i in range(200):
    print(i)
    idx = i%100
    feature = feature_list[idx].float().cuda()
    target[i*100:(i+1)*100] = i%100
    #feature = feature[:1000]
    #start_time = time.time()
    negative_samples = new_dis.rsample((4000,))
    outliers = generate_outliers(ID=feature, input_index=index,negative_samples=negative_samples, K=100, sample_number=100, select=10,cov_mat=0.1, sampling_ratio=1.0)
    outliers_list.append(outliers.cpu())
    #end_time = time.time()
    #print("time cost:", float(end_time - start_time) * 1000.0, "ms")
outliers_list = torch.cat(outliers_list, dim=0)
print(outliers_list.shape)
torch.save(outliers_list, '/nobackup-slow/taoleitian/CLIP_visual_feature/OOD/OOD_10/feature.pt')
torch.save(target, '/nobackup-slow/taoleitian/CLIP_visual_feature/OOD/OOD_10/target.pt')


