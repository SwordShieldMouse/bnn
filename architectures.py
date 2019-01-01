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

class Net(nn.Module):
    # for classifying MNIST digits
    # first value of sizes is input size
    # last value of sizes is the output size
    def __init__(self, sizes):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.leaky_relu = nn.LeakyReLU()

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        out = self.leaky_relu(self.layers[0](x))
        if len(self.layers) > 1:
            for i in range(1, len(self.layers) - 1):
                out = self.leaky_relu(self.layers[i](out))
        return self.logsoftmax(self.layers[len(self.layers) - 1](out))

class BNN(nn.Module):
    def __init__(self, sizes):
        super(BNN, self).__init__()

        self.net = Net(sizes)
        self.sizes = sizes

    def model(self, x, y):
        priors = {}
        for i in range(len(self.net.layers)):
            # assume priors are normal, but could change
            w_prior = dist.Normal(loc = torch.zeros_like(self.net.layers[i].weight), scale = torch.ones_like(self.net.layers[i].weight))
            b_prior = dist.Normal(loc = torch.zeros_like(self.net.layers[i].bias), scale = torch.ones_like(self.net.layers[i].bias))
            priors['layers.{}.weight'.format(i)] = w_prior
            priors['layers.{}.bias'.format(i)] = b_prior

        lifted_net = pyro.random_module("module", self.net, priors)
        lifted_model = lifted_net()
        with pyro.plate("map", x.shape[0]):
            probs = lifted_model(x)
            pyro.sample("obs", dist.Categorical(logits = probs).to_event(1), obs = y)

    def guide(self, x, y):
        priors = {}
        for i in range(len(self.net.layers)):
            w_loc = pyro.param("w{}_loc".format(i), torch.randn_like(self.net.layers[i].weight))
            w_scale = pyro.param("w{}_scale".format(i), torch.randn_like(self.net.layers[i].weight))
            b_loc = pyro.param("b{}_loc".format(i), torch.randn_like(self.net.layers[i].bias))
            b_scale = pyro.param("b{}_scale".format(i), torch.randn_like(self.net.layers[i].bias))
            w_prior = dist.Normal(loc = w_loc, scale = w_scale)
            b_prior = dist.Normal(loc = b_loc, scale = b_scale)
            priors['layers.{}.weight'.format(i)] = w_prior
            priors['layers.{}.bias'.format(i)] = b_prior
        lifted_net = pyro.random_module("module", self.net, priors)
