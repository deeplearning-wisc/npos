
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 2
import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'Arial'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial'

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
neg_fea = np.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/data_feature/in_score.npy', allow_pickle=True)
pos_fea = np.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/data_feature/out_score.npy', allow_pickle=True)

index = 0
for fea in pos_fea:
    if index == 0:
        pos_np = fea.cpu().data.numpy()
        index += 1
    else:
        pos_np = np.concatenate([pos_np, fea.cpu().data.numpy()], 0)

index = 0
for fea in neg_fea:
    if index == 0:
        neg_np = fea.cpu().data.numpy()
        index += 1
    else:
        neg_np = np.concatenate([neg_np, fea.cpu().data.numpy()], 0)
# breakpoint()
id_pd = pd.Series(pos_np)
# # id_pd.rename('ID')
#
ood_pd = pd.Series(neg_np)
# # ood_pd.rename('OOD')
# # data_plot = {'Energy': np.concatenate((-id_score[0:2000], -ood_score), 0), 'label':['ID'] * len(-id_score[0:2000]) + \
# #                                                                            ['OOD'] * len(-ood_score)}
# # df_after = pd.DataFrame(data=data_plot)
# # sns.histplot(data=df_after, x="Energy", hue="label")
plt.figure(figsize=(10,8))
p1 = sns.kdeplot(id_pd, shade=True, color="#4C3C83", label='ID',linewidth=2.5)
p1 = sns.kdeplot(ood_pd, shade=True, color="#E29C7A", label='OOD',linewidth=2)
# plt.xlabel("Negative energy score", fontsize=25)
# plt.ylabel("Density", fontsize=25)
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
plt.legend(fontsize=30, frameon=False)
plt.axis('off')
plt.savefig('ddd.jpg', dpi=500)