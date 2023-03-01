'''import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
ID = torch.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/data_feature/clip_ID_3.pt')
#outliers = torch.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/data_feature/clip_outliers_3.pt')[-200:]
ID_mean = torch.mean(ID, dim=0)
ID_cov = torch.cov(ID.T)
VOS_list = []
for i in range(200):
    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
        ID_mean, covariance_matrix=ID_cov)
    negative_samples = new_dis.rsample((10000,))
    prob_density = new_dis.log_prob(negative_samples)
    cur_samples, index_prob = torch.topk(- prob_density, 1)
    VOS_list.append(negative_samples[index_prob])
VOS_list = torch.cat(VOS_list, dim=0)
outliers = VOS_list
torch.save(outliers, 'VOS.pt')
plt.figure(dpi=300)
#outliers = ID
X = torch.cat((ID, outliers), dim=0)
#X = ID
X = X.numpy()
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X).T

plt.scatter(X_tsne[0,:ID.shape[0]], X_tsne[1,:ID.shape[0]], c='#7373a2')
plt.scatter(X_tsne[0,ID.shape[0]:], X_tsne[1,ID.shape[0]:], c='#e29c7a')
plt.xticks([])
plt.yticks([])

plt.savefig(r"t-SNE_1.png")

plt.show()
print(X_tsne)'''