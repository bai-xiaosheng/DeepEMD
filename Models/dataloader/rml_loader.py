import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import argparse
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pickle

import random
import scipy.io as sio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
class radioml_loader():

    def __init__(self, args,mod='train',support = True):

        xd, snrs, mods, lbl, data, n_all_class, self.n_examples = self.RML2016a()
        # 1         128        2
        self.im_width, self.im_height, self.channels = list(map(int, args.x_dim.split(',')))  # map(a,b)对b执行a操作，并返回迭代器

        self.encoder_length = args.h_dim * 8
        self.n_query = self.n_examples - args.shot
        self.n_test_query = self.n_examples - args.shot
        self.n_class = n_all_class - args.way  # n_way:训练集
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
        # analogs_index  = [1, 2, 10]
        # digitals_index = [0, 3, 4, 5, 6, 7, 8, 9]

        analogs_index = [0, 1, 2]
        digitals_index = [3, 4, 5, 6, 7, 8, 9, 10]

        ## (11, 10000, 1, 128, 2)
        x_train = data[digitals_index, :, :, :, :]
        self.train_dataset1 = x_train.astype('float32')
        # train_dataset1.shape = (8, 20000, 1, 2, 128)

        x_test = data[analogs_index, :, :, :, :]
        self.test_dataset = x_test.astype('float32')
        # test_dataset.shape = (3, 20000, 1, 2, 128)

        ep_classes = np.random.permutation(self.n_class)[:args.way]
        self.train_dataset = self.train_dataset1[ep_classes, :, :, :, :]
        if mod == 'train':
            selected1 = np.random.permutation(self.n_examples)[:args.shot]
            selected2 = np.arange(self.n_examples)
            selected3 = np.delete(selected2, selected1)
            if support is True:
                self.data = self.train_dataset[:,selected1,:,:,:]
                self.label = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.shot)).astype(np.uint8).reshape(-1)
            else:
                self.data = self.train_dataset[:,selected3,:,:,:]
                self.label = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.query)).astype(np.uint8).reshape(-1)
        elif mod == 'test':
            selected1 = np.random.permutation(self.n_examples)[:args.shot]
            selected2 = np.arange(self.n_examples)
            selected3 = np.delete(selected2, selected1)
            if support is True:
                self.data = self.test_dataset[:,selected1,:,:,:]
                self.label = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.shot)).astype(np.uint8).reshape(-1)
            else:
                self.data = self.test_dataset[:,selected3,:,:,:]
                self.label = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.query)).astype(np.uint8).reshape(-1)
        self.data = np.reshape(self.data, (self.data.shape[0]*self.data.shape[1],) + self.data.shape[2:])
        self.data = torch.from_numpy(np.transpose(self.data,(0,3,1,2)))
        self.label = torch.from_numpy(self.label)
        self.label = self.label.type(torch.LongTensor)
        # set environment variables: gpu, num_thread
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        torch.set_num_threads(2)  # 设置pytorch在cpu中并行计算时所占用的线程数

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data.to(self.device),label.to(self.device)

    def __len__(self):
        return len(self.data)


    # def train_loader(self,args,):
    #
    #
    #     support = np.zeros([args.way, args.shot, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #     query = np.zeros([args.way, self.n_query, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #
    #     for c in range(args.way):
    #         selected1 = np.random.permutation(self.n_examples)[:args.shot]
    #         selected2 = np.arange(self.n_examples)
    #         selected3 = np.delete(selected2, selected1)
    #
    #         support[c, :, :, :, :] = self.train_dataset[c, selected1, :, :, :]
    #         query[c, :, :, :, :] = self.train_dataset[c, selected3, :, :, :]
    #
    #     s_labels = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.shot)).astype(np.uint8)
    #     q_labels = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.query)).astype(np.uint8)
    #
    #     # sample data for next batch
    #     support = np.reshape(support, (support.shape[0] * support.shape[1],) + support.shape[2:])
    #     # support.shape = (15, 60, 60, 4)
    #
    #     support = torch.from_numpy(np.transpose(support, (0, 3, 1, 2)))
    #     # torch.Size([15, 4, 60, 60])
    #
    #     query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
    #     # query.shape = (585, 60, 60, 4)
    #
    #     query = torch.from_numpy(np.transpose(query, (0, 3, 1, 2)))
    #     # query.shape = torch.Size([585, 4, 60, 60])
    #
    #     s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
    #     # s_labels.shape = torch.Size([15])
    #
    #     q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
    #     # q_labels.shape = torch.Size([585])
    #
    #     s_labels = s_labels.type(torch.LongTensor)
    #     # s_labels.shape = torch.Size([15])
    #
    #     q_labels = q_labels.type(torch.LongTensor)
    #     # q_labels.shape = torch.Size([585])
    #
    #     s_onehot = torch.zeros(args.way * args.shot, args.way).scatter_(1, s_labels.reshape(-1, 1), 1)
    #     # s_onehot.shape = torch.Size([15, 3])
    #
    #     q_onehot = torch.zeros(args.way * args.query, args.way).scatter_(1, q_labels.reshape(-1, 1), 1)
    #     # q_onehot.shape = torch.Size([585, 3])
    #
    #     # inputs = [support.to(device), s_onehot.to(device), query.to(device), q_onehot.to(device)]
    #     return support,s_labels.to(self.device),query,q_labels.to(self.device)
    # def pretrain_loader(self,args,):
    #
    #     data_pre = np.zeros([args.way,self.n_query+ args.shot, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #     # query = np.zeros([args.way, self.n_query, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #
    #     for c in range(args.way):
    #         # selected1 = np.random.permutation(self.n_examples)[:args.shot]
    #         selected2 = np.arange(self.n_examples)
    #         # selected2 = np.arange(1000)
    #         # selected3 = np.delete(selected2, selected1)
    #
    #         data_pre[c, :, :, :, :] = self.train_dataset[c, selected2, :, :, :]
    #         # query[c, :, :, :, :] = self.train_dataset[c, selected3, :, :, :]
    #     data_label = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.shot+self.n_query)).astype(np.uint8)
    #     data_pre = np.reshape(data_pre, (data_pre.shape[0] * data_pre.shape[1],) + data_pre.shape[2:])
    #     # # support.shape = (15, 60, 60, 4)
    #     #
    #     # support = torch.from_numpy(np.transpose(support, (0, 3, 1, 2)))
    #     data_pre = torch.from_numpy(np.transpose(data_pre, (0, 3, 1, 2)))
    #     # # torch.Size([15, 4, 60, 60])
    #     #
    #     # query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
    #     # # query.shape = (585, 60, 60, 4)
    #     #
    #     # query = torch.from_numpy(np.transpose(query, (0, 3, 1, 2)))
    #     # # query.shape = torch.Size([585, 4, 60, 60])
    #     #
    #     # s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
    #     data_label = torch.from_numpy(np.reshape(data_label, (-1,)))
    #     # # s_labels.shape = torch.Size([15])
    #     #
    #     # q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
    #     # # q_labels.shape = torch.Size([585])
    #     #
    #     # s_labels = s_labels.type(torch.LongTensor)
    #     data_label = data_label.type(torch.LongTensor)
    #     # # s_labels.shape = torch.Size([15])
    #     #
    #     # q_labels = q_labels.type(torch.LongTensor)
    #     # # q_labels.shape = torch.Size([585])
    #     #
    #     # s_onehot = torch.zeros(args.way * args.shot, args.way).scatter_(1, s_labels.reshape(-1, 1), 1)
    #     # # s_onehot.shape = torch.Size([15, 3])
    #     #
    #     # q_onehot = torch.zeros(args.way * args.query, args.way).scatter_(1, q_labels.reshape(-1, 1), 1)
    #     # q_onehot.shape = torch.Size([585, 3])
    #
    #     # inputs = [support.to(device), s_onehot.to(device), query.to(device), q_onehot.to(device)]
    #     return data_pre.to(self.device),data_label.to(self.device)
    #
    #
    #
    # def test_loader(self,args):
    #     support = np.zeros([args.way, args.shot, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #     query = np.zeros([args.way, self.n_test_query, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #     # query = np.zeros([args.way, 1000, self.im_width, self.im_height, self.channels], dtype=np.float32)
    #
    #     for c in range(args.way):
    #         selected1 = np.random.permutation(self.n_examples)[:args.shot]
    #         selected2 = np.arange(self.n_examples)
    #         selected3 = np.delete(selected2, selected1)
    #         # selected3 =np.arange(1000)
    #
    #         support[c, :, :, :, :] = self.test_dataset[c, selected1, :, :, :]
    #         query[c, :, :, :, :] = self.test_dataset[c, selected3, :, :, :]
    #
    #     s_labels = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.shot)).astype(np.uint8)
    #     q_labels = np.tile(np.arange(args.way)[:, np.newaxis], (1, args.query)).astype(np.uint8)
    #
    #     # sample data for next batch
    #
    #     support = np.reshape(support, (support.shape[0] * support.shape[1],) + support.shape[2:])
    #     # support.shape = (15, 60, 60, 4)
    #
    #     support = torch.from_numpy(np.transpose(support, (0, 3, 1, 2)))
    #     # support.shape = torch.Size([15, 4, 60, 60])
    #
    #     query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
    #     # query.shape = (807, 60, 60, 4)
    #
    #     query = torch.from_numpy(np.transpose(query, (0, 3, 1, 2)))
    #     # query.shape = torch.Size([807, 4, 60, 60])
    #
    #     s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
    #     # s_labels.shape = torch.Size([15])
    #
    #     q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
    #     # q_labels.shape = torch.Size([807])
    #
    #     s_labels = s_labels.type(torch.LongTensor)
    #     # s_labels.shape = torch.Size([15])
    #
    #     q_labels = q_labels.type(torch.LongTensor)
    #     # q_labels.shape = torch.Size([807])
    #
    #     s_onehot = torch.zeros(args.way * args.shot, args.way).scatter_(1, s_labels.reshape(-1, 1), 1)
    #     # s_onehot.shape = torch.Size([15, 3])
    #
    #     q_onehot = torch.zeros(args.way * args.query, args.way).scatter_(1, q_labels.reshape(-1, 1), 1)
    #     return support.to(self.device),s_labels.to(self.device),query,q_labels.to(self.device)

    def RML2016a(self):
        xd = pickle.load(open("D:/DeepLearning/DeepEMD-master/data/RML2016.10a_dict.pkl", 'rb'), encoding='iso-8859-1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0])

        # del snrs[0:18]
        x = []
        lbl = []
        # selected1 = np.random.permutation(self.n_examples)[:args.shot]
        del snrs[0:10]

        for mod in mods:
            for snr in snrs:
                x.append(xd[(mod, snr)])
                for i in range(xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        x = np.vstack(x)  # 拼接

        n_all_class = len(mods)
        n_per_class = np.array(x.shape[0] / n_all_class, dtype=np.int)

        # x = x.reshape((-1, 256))
        # scaler = sklearn.preprocessing.MinMaxScaler()
        # x = scaler.fit_transform(x)

        x = x.reshape((n_all_class, n_per_class, 1, 2, 128))

        x = np.transpose(x, (0, 1, 2, 4, 3))
        # x = x[:,:1000,:,:,:]
        # x1 = np.linspace(0,127,128)
        # a0 = y[9,19999,:,:,0].reshape(128,)
        # a1 = y[9,19999,:,:,1].reshape(128,)
        # plt.plot(x1, a0, color = 'red')
        # plt.plot(x1, a1, color = 'blue')
        #       所有类  正数  十一类 标签 数据  类别：十一类 每类个数：一万
        return xd, snrs, mods, lbl, x, x.shape[0], x.shape[1]

