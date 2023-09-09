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
    def __init__(self, Dataset, Path, model, checkpoint, task):
        ## task
        self.task = task 

        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=checkpoint, mode='test')
        self.data   = Dataset.Data(self.cfg, model)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=6)

        self.net = get_model(self.cfg, model)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                # out1, out2, out3, out4, pose = self.net(image, shape, name)
                out1, out2, out3, out4 = self.net(image, shape, name)
                pred = torch.sigmoid(out4[0,0]).cpu().numpy()*255
                head = '/home/biyu/ICON-main/util/evaltool/Prediction/'+model+'/'+ self.cfg.datapath.split('/')[-2]
                if not os.path.exists(head):
                    print("create a new folder: {}".format(head))
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # ICON-V: VGG16, ICON-R: ResNet50, ICON-S: Swin384_22k, ICON-P: PVTv2, CycleMLP: B4
    parser.add_argument("--model", default='ICON-P')
    # Tasks: SOD, SOC-Attr, COD, FPS
    parser.add_argument("--task", default='SOD')
    parser.add_argument("--ckpt", default='/home/biyu/ICON-main/checkpoint/ICON/ICON-P/ICON-0707-P101')
    
    args   = parser.parse_args()
    task   = args.task
    model  = args.model
    ckpt   = args.ckpt
    
    print(args.model, args.ckpt)

    if args.task == "SOD":
        for path in ['/home/biyu/ICON-main/datasets/ECSSD/Test', '/home/biyu/ICON-main/datasets/PASCAL-S/Test', '/home/biyu/ICON-main/datasets/DUTS/Test', '/home/biyu/ICON-main/datasets/HKU-IS/Test', '/home/biyu/ICON-main/datasets/DUT-OMRON/Test', '/home/biyu/ICON-main/datasets/SOD/Test']:
            t = Test(dataset, path, model, ckpt, task)
            t.save()

    elif args.task == "SOC":
        for path in ['/home/biyu/ICON-main/datasets/SOC/SOC-AC', '/home/biyu/ICON-main/datasetsSOC/SOC-BO', '/home/biyu/ICON-main/datasets/SOC/SOC-CL', '/home/biyu/ICON-main/datasets/SOC/SOC-HO', '/home/biyu/ICON-main/datasetsSOC/SOC-MB', '/home/biyu/ICON-main/datasets/SOC/SOC-OC', '/home/biyu/ICON-main/datasets/SOC/SOC-OV', '/home/biyu/ICON-main/datasets/SOC/SOC-SC', '/home/biyu/ICON-main/datasets/SOC/SOC-SO']:
            t = Test(dataset, path, model, ckpt, task)
            t.save()

    elif args.task == "COD":
        for path in ['/home/biyu/ICON-main/datasets/CHAMELEON/Test', '/home/biyu/ICON-main/datasets/CAMO/Test', '/home/biyu/ICON-main/datasetsCOD10K/Test', '/home/biyu/ICON-main/datasets/CPD1K/Test']:
            t = Test(dataset, path, model, ckpt)
            t.save()

    else:
        # For testing FPS
        inf_time = 0
        for path in ['/home/biyu/ICON-main/datasets/SOD/Test/']:
            start = datetime.datetime.now()
            start = time.time()
            t = Test(dataset, path, model, ckpt)
            end = datetime.datetime.now()
            t.save()
            end = time.time()
            inf_time += (end - start)
        inf_per_image = inf_time / 300
        fps = 1 /  inf_per_image
        print("inf_per_image: {}, fps: {}".format(inf_per_image, fps))
