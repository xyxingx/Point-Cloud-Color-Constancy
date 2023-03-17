from __future__ import print_function
import os
import sys
import argparse
import random
import json
import time
import datetime
from pointnet.model import PCCC
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from utils_p import *
# import tqdm

sys.path.append('./pointnet/')
from model import *
from dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--lrate', type=float, default=0.0003, help='learning rate')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--foldnum', type=int, default=1, help='fold number')
parser.add_argument('--sizes', type=int, default=16, help='size_scale_16_or_64')
parser.add_argument('--gpu_ids', type=str, default='0', help='choice a gpu')
parser.add_argument('--datasets', type=str, default='ETH3D', help='select dataset')

opt = parser.parse_args()
print(opt)
log_name = opt.datasets+'_'+str(opt.sizes)
log_path = os.path.join('./log/',log_name)
event_path = os.path.join('./envent/',log_name+'_'+str(opt.foldnum))
if not os.path.exists(log_path):
    os.makedirs(log_path)
# if not os.path.exists(event_path):
#     os.makedirs(event_path)

# visualization
# writer = SummaryWriter(event_path)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
train_loss = AverageMeter()

# load data
dataset_train = PcdColor(train=True,foldn=opt.foldnum,sizes=opt.sizes)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.workers)
len_dataset_train = len(dataset_train)
print('len_dataset_train:', len(dataset_train))
dataset_test = PcdColor(train=False,foldn=opt.foldnum,sizes=opt.sizes)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=opt.workers)
len_dataset_test = len(dataset_test)
print('len_dataset_test:', len(dataset_test))
# print('training fold %d' % opt.foldnum)

# create network
PointsNet = PCCC(k=3,sis=opt.sizes**2) 
network = PointsNet.cuda()

# if opt.pth_path != '':
#     print('loading pretrained model')
#     network.load_state_dict(torch.load(opt.pth_path))
print(PointsNet)
logname = os.path.join(log_path,'log_'+str(opt.foldnum)+'.txt')
with open(logname, 'a') as f:
    f.write(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'\n')
    f.write('fold:'+str(opt.foldnum)+'\n')
    f.write(str(network) + '\n')

# optimizer
lrate = opt.lrate
optimizer = optim.Adam(network.parameters(), lr=lrate)

train_loss = AverageMeter()
val_loss = AverageMeter()
# train
print('start train.....')
best_val_loss = 999999
step = 0
for epoch in range(opt.nepoch):
    # train mode
    time_use1 = 0
    train_loss.reset()
    network.train()
    start = time.time()
    for i, data in enumerate(dataloader_train):
        optimizer.zero_grad()
        img, label, fn = data
        img = img.cuda()
        label = label.cuda()
        pred,_ = network(img)
        loss = get_angular_loss(torch.sum(pred,2), label)
        print('Inter:%d,loss:%f'%(i,loss))
        loss.backward()
        train_loss.update(loss.item())
        optimizer.step()
        # writer.add_scalar('train_loss', loss,step )
        step = step+1
    time_use1 = time.time() - start
##        val mode
    time_use2 = 0
    val_loss.reset()
    with torch.no_grad():
        if epoch % 1 == 0:
            val_loss.reset()
            network.eval()
            start = time.time()
            errors = []
            for i, data in enumerate(dataloader_test):
                img, label, fn = data
                img = img.cuda()
                label = label.cuda()
                pred,_ = network(img)
                loss = get_angular_loss(torch.sum(pred, 2), label)
                val_loss.update(loss.item())
                errors.append(loss.item())
                # writer.add_scalar('test_loss', loss, epoch)
            time_use2 = time.time() - start
                # print('visdom error......')
            mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
    try:
        print('Epoch: %d,  Train_loss: %f,  Val_loss: %f, T_Time: %f, V_time: %f' %(epoch, train_loss.avg, val_loss.avg, time_use1, time_use2))
    except:
        print('IOError...')
    if (val_loss.avg > 0 and val_loss.avg < best_val_loss):
        best_val_loss = val_loss.avg
        best_mean = mean
        best_median = median
        best_trimean = trimean
        best_bst25 = bst25
        best_wst25 = wst25
        best_pct95 = pct95
        torch.save(network.state_dict(), '%s/fold%d.pth' % (log_path, opt.foldnum))
    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "best_val_loss": best_val_loss,
        "mean": best_mean,
        "median": best_median,
        "trimean": best_trimean,
        "bst25": best_bst25,
        "wst25": best_wst25,
        "pct95": best_pct95
    }
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
