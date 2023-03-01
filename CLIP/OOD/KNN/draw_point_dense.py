import torch
import faiss
import time
from torch.distributions import MultivariateNormal
import faiss.contrib.torch_utils
from KNN import generate_outliers, generate_outliers_OOD
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

feature_list = torch.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/data_dict.pt')
outliers_list = []
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 342)
new_dis = MultivariateNormal(torch.zeros(342),  torch.eye(342))

feature = feature_list.float().cuda()
for i in range(1):
    print(i)
    start_time = time.time()
    outliers = generate_outliers_OOD(ID=feature, input_index=index,negative_samples=feature,select=200, K=300)
    outliers_list.append(outliers.cpu())
    end_time = time.time()
    print("time cost:", float(end_time - start_time) * 1000.0, "ms")
outliers_list = torch.cat(outliers_list, dim=0).cuda()
mean = outliers_list.mean(0)
var = torch.cov(outliers_list.T)
#distribute= MultivariateNormal(torch.zeros(512).cuda(),  torch.eye(512).cuda())
distribute= MultivariateNormal(mean, var*2+0.00001*torch.eye(342).cuda())
output = []
for i in range(60):
    negative_samples = distribute.rsample((10000,))
    prob_density = distribute.log_prob(negative_samples)
    #index_prob = (prob_density > 520).nonzero().view(-1)
    cur_samples, index_prob = torch.topk(- prob_density, 1)
    #avg_point = torch.mean(avg, dim=0)
    #negative_samples = mean + torch.mm(negative_samples, var*3)
    #outliers = generate_outliers_OOD(ID=feature, input_index=index,negative_samples=negative_samples,select=5, K=100)
    outliers = negative_samples[index_prob]
    output.append(outliers)
output = torch.cat(output, dim=0)
print(output.shape)
output = outliers_list
#torch.save(outliers_list, '/nobackup-slow/taoleitian/CLIP_visual_feature/OOD/OOD_10/feature.pt')
#torch.save(target, '/nobackup-slow/taoleitian/CLIP_visual_feature/OOD/OOD_10/target.pt')
plt.figure(dpi=200)
X = torch.cat((feature, output), dim=0)
X = X.cpu().numpy()
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X).T

plt.scatter(X_tsne[0,:feature_list.shape[0]], X_tsne[1,:feature_list.shape[0]])
plt.scatter(X_tsne[0,feature_list.shape[0]:feature_list.shape[0]+output.shape[0]], X_tsne[1,feature_list.shape[0]:feature_list.shape[0]+output.shape[0]], c='r')
#plt.scatter(X_tsne[0,feature_list.shape[0]+outliers_list.shape[0]:], X_tsne[1,feature_list.shape[0]+outliers_list.shape[0]:], c='g')
plt.show()
plt.savefig(r"r.png")
print(X_tsne)

