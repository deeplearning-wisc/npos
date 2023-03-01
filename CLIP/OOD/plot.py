from sklearn.calibration import CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt
postive =np.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/in_score_0_01.npy')
negtive =np.load('/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/vos/classification/CIFAR/out_score_0_01.npy')
x = np.concatenate((postive, negtive))
output = (x-np.min(x))/(np.max(x)-np.min(x))
label = np.concatenate((np.ones_like(postive), np.zeros_like(negtive)))
disp = CalibrationDisplay.from_predictions(label, output, n_bins=20)
plt.show()