#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import numpy as np
import torch
import argparse
import dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from model.get_model import get_model


class Test(object):
    def __init__(self, Dataset, checkpoint):
        self.cfg = Dataset.Config(datapath="", snapshot=checkpoint, mode='test')
        self.net = get_model(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self, input_folder, output_folder):
        with torch.no_grad():
            image_names = os.listdir(input_folder)
            for name in image_names:
                image_path = os.path.join(input_folder, name)
                image = cv2.imread(image_path)
                H, W, _ = image.shape
                image = image.transpose(2, 0, 1)  # Convert to channel-first format
                image = np.expand_dims(image, axis=0)
                image = torch.from_numpy(image).cuda().float()

                # Run the model on the input image
                with torch.no_grad():
                    out1, out2, out3, out4 = self.net(image, (H, W), [name])

                # Post-process the output and save the result
                pred = torch.sigmoid(out4[0, 0]).cpu().numpy() * 255
                pred_path = os.path.join(output_folder, name + '.png')
                cv2.imwrite(pred_path, np.round(pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default='../checkpoint/SCL/100')
    args = parser.parse_args()

    ckpt = args.ckpt

    # Replace 'dataset' with the actual name of the dataset module you are using.
    # For example, if the dataset module is called 'custom_dataset', you should use:
    # from custom_dataset import dataset
    t = Test(dataset, ckpt)

    # Input folder containing images for testing
    input_folder = '../util/pred/input'

    # Output folder where the predictions will be saved
    output_folder = '../util/pred/output'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    t.save(input_folder, output_folder)
