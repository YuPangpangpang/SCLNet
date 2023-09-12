#!/usr/bin/python3
#coding=utf-8
from model.SCL.SCL import SCL

def get_model(cfg, img_name=None):


    if img_name:
        model = SCL(cfg, img_name).cuda()
    else: 
        model = SCL(cfg).cuda()

    print("Model based on {} have {:.4f}Mb paramerters in total".format('SCL', sum(x.numel()/1e6 for x in model.parameters())))

    return model.cuda()
