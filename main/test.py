#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import argparse
import dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from model.get_model import get_model
import datetime
import time

class Test(object):
    def __init__(self, Dataset, Path, checkpoint):

        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=checkpoint, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=6)

        self.net = get_model(self.cfg,)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                # out1, out2, out3, out4, pose = self.net(image, shape, name)
                out1, out2, out3, out4 = self.net(image, shape, name)
                pred = torch.sigmoid(out4[0,0]).cpu().numpy()*255
                head = '/home/biyu/SCLNet/util/evaltool/Prediction/'+self.cfg.datapath.split('/')[-2]
                if not os.path.exists(head):
                    print("create a new folder: {}".format(head))
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default='../checkpoint/SCL/100')
    
    args   = parser.parse_args()
    ckpt   = args.ckpt
    
    print(args.ckpt)

    for path in ['../datasets/ECSSD/Test', '../datasets/PASCAL-S/Test', '../datasets/DUTS/Test', '../datasets/HKU-IS/Test', '../datasets/DUT-OMRON/Test', '../datasets/SOD/Test']:
        t = Test(dataset, path, ckpt)
        t.save()


