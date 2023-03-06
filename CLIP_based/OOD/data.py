import numpy as np
import random

idx = np.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/KNN-OOD-Tao/OOD/ind.npy')
path = '/nobackup-fast/ImageNet-1k_CLIP/path/10_val.txt'
file = open(path, 'r')
path_list = file.read()
path_list = eval(path_list)
cc = list(zip(idx, path_list))
random.shuffle(cc)
idx[:], path_list[:] = zip(*cc)
path_save = path_list[1700:]
for i in range(1700):
    if idx[i] == True:
        path_save.append(path_list[i])
print(len(path_save))
file = open('10_val.txt', 'w')
file.write(str(path_save))
file.close()
