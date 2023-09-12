#!/usr/bin/python3
#coding=utf-8

import os
import sys
import datetime
import dataset
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from apex import amp
#from config import param as option
from model.get_model import get_model
from test import Test
from torchvision import transforms
import matplotlib as plt

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# IoU Loss
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


# Structure Loss
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


# F Loss
def f_loss(pred, mask, beta=0.3, log_like=False):
    eps = 1e-10
    n = n = pred.size(0)
    tp = (pred * mask).view(n, -1).sum(dim=1)
    h = beta * mask.view(n, -1).sum(dim=1) + pred.view(n, -1).sum(dim=1)
    fm = (1+beta) * tp / (h+eps)
    if log_like:
        floss = -torch.log(fm)
    else:
        floss = (1-fm)

    return floss.mean()


def train(Dataset, parser):
    
    args   = parser.parse_args()
    _DATASET_ = args.dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _LOSS_ = args.loss
    _SAVEPATH_ = args.savepath
    _VALID_ = args.valid

    print(args)

    cfg = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_,
                             momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)

    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=6)
    ## network

    # danka
    net = get_model(cfg)
    net.train(True)
    net.cuda()

    base, head = [], []
    
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)     
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw = SummaryWriter(cfg.savepath)
    global_step = 0
 

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr


        for step, (image1, mask, image2) in enumerate(loader):
            image1, mask, image2 = image1.cuda(), mask.cuda(), image2.cuda()
            image = torch.concat([image1, image2], dim=0)
            bsz = mask.size(0)
            out2, out3, out4, out5 = net(image)
            o1_2 = out2[0:bsz,:,:,:]
            o1_3 = out3[0:bsz,:,:,:]
            o1_4 = out4[0:bsz,:,:,:]
            o1_5 = out5[0:bsz,:,:,:]
            o2_5 = out5[bsz:bsz*2, :, :, :]
            if _LOSS_ == "CPR":
                loss1 = F.binary_cross_entropy_with_logits(o1_2, mask) + iou_loss(o1_2, mask)
                loss2 = F.binary_cross_entropy_with_logits(o1_3, mask) + iou_loss(o1_3, mask)
                loss3 = F.binary_cross_entropy_with_logits(o1_4, mask) + iou_loss(o1_4, mask)
                loss4 = F.binary_cross_entropy_with_logits(o1_5, mask) + iou_loss(o1_5, mask)
                loss5 = F.binary_cross_entropy_with_logits(o2_5, mask) + iou_loss(o2_5, mask)
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5

            optimizer.zero_grad()
            # with amp.scale_loss(loss, optimizer) as scale_loss:
            #     scale_loss.backward()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            sw.add_scalars('loss',
                           {'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item()
                            }, global_step=global_step)
            if step%100 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss1.item(), loss2.item(), loss3.item(), loss4.item()))

        if epoch >= 59 and (epoch + 1) % 10 == 0 :
            torch.save(net.state_dict(), cfg.savepath+'/'+str(epoch+1))


if __name__=='__main__':
    ##register_amp_float_function
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='../datasets/DUTS/Train')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momen", type=float, default=0.9)  
    parser.add_argument("--decay", type=float, default=1e-4)  
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=100)
    # CPR: IoU+BCE, STR: Structure Loss, FL: F-measure loss
    parser.add_argument("--loss", default='CPR')  
    parser.add_argument("--savepath", default='../checkpoint/SCL/')
    parser.add_argument("--valid", default=True)
    train(dataset, parser)
