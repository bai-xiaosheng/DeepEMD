import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, data,label, n_batch, n_per):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.data = data.view(3, -1, 2, 1, 128)
        self.n_per = n_per
        self.label = label.view(3,-1)

        # label = np.array(label)#all data label
        # self.m_ind = []#the data index of each class
        # for i in range(max(label) + 1):
        #     ind = np.argwhere(label == i).reshape(-1)# all data index of this class
        #     ind = torch.from_numpy(ind)
        #     self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    # def iter(self):
    #     # for i_batch in range(self.n_batch):
    #     pos = torch.randperm(self.data.shape[1])[:self.n_per]
    #     data = self.data[:,pos,:,:,:]
    #     data = np.reshape(data, (data.shape[0] * data.shape[1],) + data.shape[2:])
    #     label = self.label[:,pos]
    #     label = label.view(-1)
    #         # batch = []
    #         # classes = torch.randperm(len(self.m_ind))[:self.n_cls]#random sample num_class indexs,e.g. 5
    #         # classes = range(3)
    #         # for c in classes:
    #             # l = self.m_ind[c]#all data indexs of this class
    #             # pos = torch.randperm(len(l))[:self.n_per] #sample n_per data index of this class
    #             # batch.append(l[pos])
    #         # batch = torch.stack(batch).t().reshape(-1)
    #         # .t() transpose,
    #         # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
    #         # instead of aaaabbbbccccdddd
    #     return data,label

    def __iter__(self):
        for i_batch in range(self.n_batch):
            pos = torch.randperm(self.data.shape[1])[:self.n_per]
            data = self.data[:,pos,:,:,:]
            data = np.reshape(data, (data.shape[0] * data.shape[1],) + data.shape[2:])
            label = self.label[:,pos]
            label = label.view(-1)
            yield data,label

