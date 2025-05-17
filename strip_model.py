import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
import models
path = ""
name = ""

def load_model():
    checkpoint = torch.load(path + name, weights_only=False)
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint['gen']
    torch.save(path + "s_" + name, model_name)  
    
load_model()