import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
import models
import sys
import os


def strip_model(path,name):
    checkpoint = torch.load(path + name, weights_only=False)
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint['gen']
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    model = {'model' : model, 'epoch' : epoch}
    torch.save(model, path + "strip/s_" + name)  

if __name__ == '__main__':
    path = sys.argv[1]
    if not os.path.exists(path):
        exit()
    out_path = os.path.join(path, "strip/")
    if not os.path.exists(out_path):
       os.makedirs(out_path)

    for f in sorted(os.listdir(path)):
        if f.endswith('.pth'):
            strip_model(path, f)
            