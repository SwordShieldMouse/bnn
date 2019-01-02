import torch
import pyro
import os
import numpy as np
import pyro.distributions as dist
import torch.nn as nn
import torchvision
import pyro.optim

from torch.distributions import constraints

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

class Net(nn.Module):
    # for classifying MNIST digits
    # first value of sizes is input size
    # last value of sizes is the output size
    def __init__(self, sizes):
        super(Net, self).__init__()
        # try convolutional architecture as well

        self.layers = nn.ModuleList()
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.leaky_relu = nn.LeakyReLU()

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        self.test = nn.Linear(28 * 28, 50)

    def forward(self, x):
        #print("starting forward")
        out = x
        for i in range(len(self.layers) - 1):
            #print("layer {}".format(i))
            #print(out.shape)
            out = self.leaky_relu(self.layers[i](out))
            #print(self.layers[i])
            #print(out.shape)
        return self.logsoftmax(self.layers[-1](out))

class BNN(nn.Module):
    def __init__(self, sizes):
        super(BNN, self).__init__()

        self.net = Net(sizes)
        self.sizes = sizes

    def model(self, x, y):
        priors = {}
        for i in range(len(self.net.layers)):
            # assume priors are normal, but could change
            w_size =  list(self.net.layers[i].weight.shape)
            b_size =  list(self.net.layers[i].bias.shape)
            #print(w_size)
            w_prior = dist.Normal(loc = torch.zeros(w_size), scale = torch.ones(w_size)).to_event(2)
            #print("batch dim is {}".format(w_prior.batch_shape))
            b_prior = dist.Normal(loc = torch.zeros(b_size), scale = torch.ones(b_size)).to_event(1)
            priors['layers.{}.weight'.format(i)] = w_prior
            priors['layers.{}.bias'.format(i)] = b_prior

        #print(x)
        #print(priors)
        lifted_net = pyro.random_module("module", self.net, priors)
        lifted_model = lifted_net()
        #print(y)
        with pyro.plate("map", x.shape[0]):
            probs = lifted_model(x)
            #print(probs)
            pyro.sample("obs", dist.Categorical(logits = probs), obs = y)

    def guide(self, x, y):
        priors = {}
        for i in range(len(self.net.layers)):
            w_size = list(self.net.layers[i].weight.shape)
            b_size = list(self.net.layers[i].bias.shape)

            w_loc = pyro.param("w{}_loc".format(i), torch.randn(w_size))
            w_scale = pyro.param("w{}_scale".format(i), torch.exp(torch.randn(w_size)))#, constraint = constraints.positive)
            b_loc = pyro.param("b{}_loc".format(i), torch.randn(b_size))
            b_scale = pyro.param("b{}_scale".format(i), torch.exp(torch.randn(b_size)))# constraint = constraints.positive)

            w_prior = dist.Normal(loc = w_loc, scale = w_scale).to_event(2)#.to_event(1)
            #print("batch dim is {}".format(w_prior.batch_shape))
            b_prior = dist.Normal(loc = b_loc, scale = b_scale).to_event(1)#.to_event(1)

            priors['layers.{}.weight'.format(i)] = w_prior
            priors['layers.{}.bias'.format(i)] = b_prior
        #with pyro.plate("map", x.shape[0]):
        lifted_net = pyro.random_module("module", self.net, priors)
        return lifted_net()

    def predict(self, x, num_samples = 1000):
        posterior_models = [self.guide(None, None) for _ in range(num_samples)]
        y_preds = [model(x).data for model in posterior_models]
        means = torch.mean(torch.stack(y_preds), 0)
        medians = np.percentile(torch.stack(y_preds).numpy(), 50, axis = 0)
        return np.amax(means.numpy(), axis = 1), np.argmax(means.numpy(), axis = 1), np.amax(medians, axis = 1)
