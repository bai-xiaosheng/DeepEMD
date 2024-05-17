import scipy.io as io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

C = io.loadmat('D:/DeepLearning/DeepEMD-master/checkpoint/mstar/fcn/1shot-3way/exp_3/9.mat')
c = C['C']
print('c is\n',c)
N = (c.T/np.sum(c,1)).T
plt.figure('number')
sns.heatmap(c,annot=True, fmt='.4g',cmap="Blues")
plt.figure('%')
sns.heatmap(N,annot=True, fmt='.4g',cmap="Blues")
plt.show()