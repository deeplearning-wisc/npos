import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal

np.random.seed(1)
torch.manual_seed(1)
t1 = 5 + torch.Tensor([2, 2]).view(-1,1) * torch.randn([2, 800])
t2 = 10 + torch.Tensor([1, 1]).view(-1,1) * torch.randn([2, 100])
t3 = 15 + torch.Tensor([2, 2]).view(-1,1) * torch.randn([2, 800])
t = torch.cat((t1, t2, t3), dim=1)
torch.save(t, 'tao.pt')
#t = torch.load('tao.pt')
plt.scatter(t[0], t[1])
#plt.scatter(X_tsne[0,ID.shape[0]:], X_tsne[1,ID.shape[0]:], c='r')
mean = torch.mean(t ,dim=1)
cov = torch.cov(t)

new_dis = MultivariateNormal(mean, cov)

# breakpoint()
# index_prob = (prob_density < - self.threshold).nonzero().view(-1)
# keep the data in the low density area.
pt_list = []
for i in range(100):
    negative_samples = new_dis.rsample((1000,))
    prob_density = new_dis.log_prob(negative_samples)
    cur_samples, index_prob = torch.topk(- prob_density, 1)
    pt = negative_samples[index_prob]
    pt_list.append(pt)
pt_list = torch.cat(pt_list, dim=0).T
plt.scatter(pt_list[0], pt_list[1], c='r')
plt.savefig(r"VOS.png")
plt.show()


