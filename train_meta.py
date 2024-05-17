import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import DeepEMD
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
from mstarsoc import MSTARSOC
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as scio
from datetime import datetime
from Models.dataloader.rml_loader import radioml_loader
import matplotlib.pyplot as plt
from utils.util import create_file,save_model

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
# PRETRAIN_DIR='./output/radioml/fcn/1shot-3way/exp_2/max_acc.pth'
PRETRAIN_DIR='./output/radioml/pre_train/16-0.0050-1000-0.20/max_acc.pth'
# DATA_DIR='/home/zhangchi/dataset'
DATA_DIR='./data'

parser = argparse.ArgumentParser()
# model params
parser.add_argument('-x_dim', type=str, default="1,128,2", metavar='XDIM',help='input image dims')
parser.add_argument('-h_dim', type=int, default=8, metavar='HDIM',help="dimensionality of hidden layers (default: 64)")
parser.add_argument('-z_dim', type=int, default=8, metavar='HDIM', help="dimensionality of hidden layers (default: 64)")
#about dataset and training
parser.add_argument('-dataset', type=str, default='radioml', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
parser.add_argument('-set',type=str,default='train',choices=['test','train'],help='the set used for validation')# set used for validation
#about training
parser.add_argument('-trbs', type=int, default=100,help='batch size of tasks')
parser.add_argument('-nt', type=int, default=10,help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=1)  #训练次数
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-step_size', type=int, default=300)  #10轮后更新学习率
parser.add_argument('-gamma', type=float, default=0.5)  #每次更新学习率的0.5倍
parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch') #随机抽样任务，store_true当触发action时为True
parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
#about task
parser.add_argument('-way', type=int, default=3)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=9999,help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=1, help='number of validation episode')
parser.add_argument('-bs', type=int, default=500,help='batch size of tasks')
parser.add_argument('-te',type=int,default=10,help='number of batch')  #
parser.add_argument('-test_episode', type=int, default=100, help='number of testing episodes after training')
# about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
#deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
#deepemd sampling only
parser.add_argument('-num_patch',type=int,default=9)
#deepemd grid only patch_list
parser.add_argument('-patch_list',type=str,default='2,3',help='the size of grids at every image-pyramid level')
parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
# slvoer about
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv', 'qpth'])
parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
parser.add_argument('-l2_strength', type=float, default=0.000001)
# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

# OTHERS
parser.add_argument('-gpu', default='0')
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)

args = parser.parse_args()
pprint(vars(args))

#transform str parameter into list
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

set_seed(args.seed)
num_gpu = set_gpu(args)
# Dataset=set_up_datasets(args)

# model
# args.pretrain_dir=osp.join(args.pretrain_dir,'max_acc.pth')
model = DeepEMD(args)
model = load_model(model, args.pretrain_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()


args.save_path = '%s/%s/%dshot-%dway/'%(args.dataset,args.deepemd,args.shot,args.way)

args.save_path=osp.join('output',args.save_path)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)
## if "THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument" error occurs on GTX 2080Ti, set the following to False
torch.backends.cudnn.benchmark = True  # 让cudnn自己选择卷积速度最快的
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
#读取数据

optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  #更新学习率，每step_size更新一次，每次更新将学习率变为原来的gamma倍

trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0



result_list=[args.save_path]
result_list.append('parameter:')
result_list.append(str(args))
result_list.append('train output:')
args.save_path,exp = create_file(args.save_path, 'exp')
# args.save_path = osp.join(args.save_path,'exp_5/')
print ('The file will save '+args.save_path)
# a =os.listdir(result_list[0])
writer = SummaryWriter(osp.join(result_list[0], 'tf/' + exp + '-' + TIMESTAMP))  # 创建事件文件，并向其中添加摘要和事件

# data_load = radioml_loader(args)
for epoch in range(1, args.max_epoch + 1):
    start_time=time.time()
    tl = Averager()
    ta = 0
    correct = 0  # correct 计数分类正确的数目
    total_loss = 0
    # pred_sum=torch.tensor([]).cuda()
    # target_sum=torch.tensor([]).cuda()

    # tqdm_gen = tqdm.tqdm(range(args.max_epoch))
    model.train()

    # data_shot,s_label, data_querys,q_labels = data_load.train_loader(args)
    support = radioml_loader(args,'train',support=True)
    support_loader = DataLoader(support,batch_size=args.bs,shuffle=True)
    for i,(data_shots,label) in enumerate(support_loader):
        model.module.mode = 'encoder'
        data_shot = model(data_shots)
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)

    query = radioml_loader(args,'train',support=False)
    query_loader = DataLoader(query,batch_size=args.bs,shuffle=True)
    for i,(data_query, q_label) in enumerate(query_loader):

        model.module.mode = 'encoder'
        data_query = model(data_query)
        data_shot = model(data_shots)
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss = F.cross_entropy(logits, q_label)
        # acc = count_acc(logits, label)
        pred = logits.max(1, keepdim=True)[1].squeeze(1)  ### 找到概率最大的下标
        correct += pred.eq(q_label).sum().item()
        # pred_sum = torch.cat((pred_sum.cuda(), pred))
        # target_sum = torch.cat((target_sum, q_label))
        total_loss += loss#batch of tasks, done by accumulate gradients
        loss.backward()
        tl.add(loss.item())
    # detect_grad_nan(model)
    total_loss = total_loss/args.nt
    # if i%args.bs==0: #批量处理任务，累计梯度
    optimizer.step()
    optimizer.zero_grad()

    ta=correct/(args.nt*args.trbs*args.way)

    # writer.add_scalar('data/loss', float(loss), global_count)

    # tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
    #       .format(epoch, total_loss.item(), ta))

    writer.add_scalar('data/acc', float(ta), epoch)
    writer.add_scalar('data/loss', float(total_loss), epoch)
    tl = tl.item()
    # ta = correct/(i*label.size(0))
    print('\ntrain epoch:{} loss = {:.6f} acc = {:.4f}'.format(epoch,total_loss,ta))
    if ta >= trlog['max_acc']:
        print ('*********A better model is found*********')
        trlog['max_acc'] = ta
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc',model,args.save_path)
    if epoch == args.max_epoch:
        save_model('last',model,args.save_path)
    # #validation
    # vl = Averager()
    # va = 0
    # correct = 0  # correct 计数分类正确的数目
    # pred_sum = torch.tensor([]).cuda()
    # target_sum = torch.tensor([]).cuda()
    # model.eval()
    # args.set = 'test'
    # with torch.no_grad():
    #     # tqdm_gen = tqdm.tqdm(args.test_episode)
    #     # for i, batch in enumerate(tqdm_gen, 1):
    #     #     data, _ = [_.cuda() for _ in batch]
    #     #     k = args.way * args.shot
    #     data_shot, s_label, data_query, q_label = data_load.test_loader(args)
    #     model.module.mode = 'encoder'
    #     data_shot = model(data_shot)
    #     data_query = model(data_query)
    #     # data_shot, data_query = data[:k], data[k:]
    #     model.module.mode = 'meta'
    #     if args.shot > 1:
    #         data_shot = model.module.get_sfc(data_shot)
    #     logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
    #
    #     loss = F.cross_entropy(logits, q_label)
    #     acc = count_acc(logits, q_label)
    #     pred = logits.max(1, keepdim=True)[1].squeeze(1)  ### 找到概率最大的下标
    #     correct += pred.eq(q_label).sum().item()
    #     pred_sum = torch.cat((pred_sum, pred))
    #     target_sum = torch.cat((target_sum, q_label))
    #     vl.add(loss.item())
    #     va=correct/len(q_label)
    #
    # # sns.set()
    # C = confusion_matrix(target_sum.cpu(), pred_sum.cpu())
    # print('C is\n', C)
    # # sns.heatmap(C, annot=True, fmt='.20g')
    #
    # dataFile = os.path.join(args.save_path, '%d.mat'%epoch)
    # # print(dataFile)
    # scio.savemat(dataFile, {'C': C})
    #
    # vl = vl.item()
    # writer.add_scalar('data/val_loss', float(vl), epoch)
    # writer.add_scalar('data/val_acc', float(va), epoch)
    # # tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    #
    # print ('val epoch:{} val acc:{:.4f}'.format(epoch,va))
    # if va >= trlog['max_acc']:
    #     print ('*********A better model is found*********')
    #     trlog['max_acc'] = va
    #     trlog['max_acc_epoch'] = epoch
    #     save_model('max_acc')

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    # trlog['val_loss'].append(vl)
    # trlog['val_acc'].append(va)

    # result_list.append('epoch:%05d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f'%(epoch,tl,ta,vl,va))
    result_list.append('epoch:%05d,training_loss:%.5f,training_acc:%.5f'%(epoch,tl,ta))

    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all:
        save_model('epoch-%d'%epoch)
        torch.save(optimizer.state_dict(), osp.join(args.save_path,'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print ('This epoch takes %d seconds'%(time.time()-start_time),'  still need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
    lr_scheduler.step()

writer.close()



# Test Phase
test_acc_record = np.zeros((args.test_episode,))
model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
# model.load_state_dict(torch.load(args.pretrain_dir)['params'])
model.eval()

ave_acc = Averager()
max_acc = 0
result_list.append('test output:')
acc_test_total = np.zeros([args.test_episode, 1], dtype=np.float32)
for epoch in range(1,args.test_episode+1):
    start_time = time.time()
    # tqdm_gen = tqdm.tqdm(loader)
    correct = 0  # correct 计数分类正确的数目
    length = 0
    pred_sum=torch.tensor([]).cuda()
    target_sum=torch.tensor([]).cuda()
    with torch.no_grad():
    # for i, batch in enumerate(tqdm_gen, 1):
    #     data, _ = [_.cuda() for _ in batch]
    #     k = args.way * args.shot
        args.set = 'test'
        support = radioml_loader(args,mod='test',support=True)
        support_loader = DataLoader(support,batch_size=args.bs,shuffle=True)
        for i,(data_shot,label) in enumerate(support_loader):
            model.module.mode = 'encoder'
            data_shot = model(data_shot)
            model.module.mode = 'meta'
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
        # for i in range(args.te):
        query = radioml_loader(args,mod='test',support=False)
        query_loader = DataLoader(query,batch_size=args.bs,shuffle=True)
        for i,(data_query,q_label) in enumerate(query_loader):
            model.module.mode = 'encoder'
            data_query = model(data_query)
            model.module.mode = 'meta'
            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            pred = logits.max(1, keepdim=True)[1].squeeze(1)  ### 找到概率最大的下标
            correct += pred.eq(q_label).sum().item()
            length += len(q_label)
            pred_sum = torch.cat((pred_sum, pred))
            target_sum = torch.cat((target_sum, q_label))
            # a = q_label.size(0)
        acc = correct / length * 100
        print('test epoch:{} acc = {:.2f}'.format(epoch, acc))
        if acc >=max_acc:
            max_acc = acc
            max_epoch = epoch
        # test_acc_record[epoch-1] = acc
        # tqdm_gen.set_description('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item(), acc))
        #     print('best test epoch:{} acc = {:.4f}'.format(epoch,max_acc))
    acc_test_total[epoch-1, :] = acc
    result_list.append('epoch:%05d, test_acc:%.2f' % (epoch, acc))
    print('This epoch takes %d seconds' % (time.time() - start_time),'  still need %.2f hour to finish' % ((time.time() - start_time) * (args.test_episode - epoch) / 3600))

    # m, pm = compute_confidence_interval(test_acc_record)

    # sns.set()
    C = confusion_matrix(target_sum.cpu(), pred_sum.cpu())
    print('C is\n', C)
    # sns.heatmap(C, annot=True, fmt='.20g')
    epo = 'test_'+str(epoch)+'.mat'
    dataFile =os.path.join(args.save_path,epo)
    print(dataFile)
    scio.savemat(dataFile, {'C': C})

result_list.append('\ntrain Best Epoch {},best train Acc {:.2f}, '.format(trlog['max_acc_epoch'], trlog['max_acc']))
result_list.append('test Best Epoch {},best test Acc {:.4f}, \n'.format(max_epoch, np.max(acc_test_total)))
result_list.append('test std:{:.5f}  max:{:.5f}  min:{:.5f}  mean{:.5f}'.format(np.std(acc_test_total),np.max(acc_test_total),np.min(acc_test_total),np.mean(acc_test_total)))
save_list_to_txt(os.path.join(args.save_path,'results.txt'),result_list)
print('std: {:.5f}'.format(np.std(acc_test_total)))
print('max: {:.5f}'.format(np.max(acc_test_total)))
print('min: {:.5f}'.format(np.min(acc_test_total)))
print('mean: {:.5f}'.format(np.mean(acc_test_total)))
# result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
# print (result_list[-2])
# print (result_list[-1])
plt.figure('2')
plt.hist(acc_test_total, 20)

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("DeepEMD Classification Accuracy on RadioML 2016.10 Alpha")
plt.savefig(args.save_path+'/%dway-%dshot'%(args.way,args.shot))
plt.show()

