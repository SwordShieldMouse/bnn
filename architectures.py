import torch
import pyro
import os
import numpy as np
import pyro.distributions as dist
import torch.nn as nn
import torchvision
import pyro.optim

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

class BNN(nn.Module):
    
