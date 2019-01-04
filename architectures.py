import torch
import pyro
import os
import numpy as np
import pyro.distributions as dist
import torch.nn as nn
import torchvision
import pyro.optim

from torch.distributions import constraints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

class Net(nn.Module):
    # for classifying MNIST digits
    # first value of sizes is input size
    # last value of sizes is the output size
    def __init__(self, sizes):
        super(Net, self).__init__()

        self.dense_size = 4 # size of dense block
        self.dense_channel_size = 7

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding = 1), # from 1 x 28 x 28 to 10 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(7, 10, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 7, 3, padding = 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride = 2), # to 10 x 14 x 14
            nn.Conv2d(7, 10, 3, padding = 1), # to 20 x 14 x 14
            nn.LeakyReLU(),
            nn.Conv2d(10, 10, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 7, 3, padding = 1), # to 20 x 14 x 14
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride = 2)#, # to 20 x 7 x 7
            #nn.Conv2d(20, 20, 1), # for dimensionality reduction
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(7, 10, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 7, 3, padding = 1),
            nn.LeakyReLU(),
            nn.FractionalMaxPool2d(2, output_size = (20, 20))
            #nn.MaxPool2d(2, stride = 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(7, 10, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 7, 3, padding = 1),
            nn.LeakyReLU(),
            nn.FractionalMaxPool2d(2, output_size = (12, 12))
            #nn.MaxPool2d(2, stride = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(7, 10, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 7, 3, padding = 1),
            nn.LeakyReLU(),
            nn.FractionalMaxPool2d(2, output_size = (7, 7))
        )

        #self.down1 = nn.Conv2d(self.dense_channel_size * self.dense_size, self.dense_channel_size, 1)
        #self.pool = nn.MaxPool2d(2, stride = 2)

        #self.dilated_conv_layers = nn.Sequential(
        #    nn.Conv2d(1, 10, kernel_size = 5, padding = 2), # from 1 x 28 x 28 to 10 x 24 x 24
        #    nn.LeakyReLU(),
        #    nn.Conv2d(10, 10, kernel_size = 3, dilation = 2), # to 10 x
        #)

        self.dense_convs1 = nn.ModuleList()
        self.dense_convs2 = nn.ModuleList()
        #self.dense_convs.append(nn.Conv2d(self.dense_channel_size, self.dense_channel_size, 3, padding = 1))
        #cumul_size = self.dense_channel_size
        #prev_size = 0
        for i in range(self.dense_size):
            self.dense_convs1.append(nn.Conv2d(self.dense_channel_size * (i + 1), self.dense_channel_size, 3, padding = 1))
            self.dense_convs2.append(nn.Conv2d(self.dense_channel_size * (i + 1), self.dense_channel_size, 3, padding = 1))
            #cumul_size += self.dense_channel_size * (i + 1)


        self.layers = nn.ModuleList()
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.leaky_relu = nn.LeakyReLU()

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))


    def forward(self, x):
        # standard convolutional layers
        #out = self.conv_layers(x)
        #out = out.view(-1, 7 * 7 * 7)

        # x is of size 1 x 28 x 28
        out = self.conv1(x) # now of size 7 * 14 * 14
        # dense block
        out = self.dense_block(out, self.dense_convs1)
        out = self.conv2(out)
        out = self.dense_block(out, self.dense_convs2)
        out = self.conv3(out)
        #out = self.pool(out) # now of size 7 * 7 * 7

        # prepare for fully-connected layer
        out = out.view(-1, 7 * 7 * 7)

        # fully-connected layers
        for i in range(len(self.layers) - 1):
            out = self.leaky_relu(self.layers[i](out))

        return self.logsoftmax(self.layers[-1](out))

    def dense_block(self, x, dense_convs):
        # implements a dense block from DenseNet
        # concatenate tensors based on channel
        xs = [x] # holds all the feature maps
        for i in range(len(dense_convs)):
            xs.append(torch.cat(xs, dim = 1))
            xs[-1] = self.leaky_relu(dense_convs[i](xs[-1]))
        return xs[-1]

class BNN(nn.Module):
    def __init__(self, sizes):
        super(BNN, self).__init__()

        self.net = Net(sizes).to(device)
        self.sizes = sizes

        if torch.cuda.is_available():
            self.cuda()

    def model(self, x, y):
        priors = {}
        # add probability model for convolutional layers as well
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
