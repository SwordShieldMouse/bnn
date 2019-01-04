import torchvision.transforms as transforms

from pyro.infer import SVI, Trace_ELBO
from architectures import *
import time

pyro.set_rng_seed(0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# dataset
train_set = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_set = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

model = BNN([7 * 7 * 7, 30, 10]).to(device)

epochs = 20
n_posterior_samples = 100
optim = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model.model, model.guide, optim, loss = Trace_ELBO())

print("starting training")
t_start = time.time()
for epoch in range(epochs):
    print("epoch {}".format(epoch))
    epoch_loss = 0
    t_epoch_start = time.time()
    for ix, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        if ix % 1000 == 0:
            print("mini-batch {} of {}".format(ix, len(train_loader.dataset) / batch_size))
        epoch_loss += svi.step(x, y)
    t_epoch_end = time.time()
    print("epoch {} loss is {}".format(epoch, epoch_loss / len(train_loader.dataset)))
    print("epoch {} took {} seconds".format(epoch, t_epoch_end - t_epoch_start))
t_end = time.time()
print("training took {} seconds".format(t_end - t_start))

print("starting evaluation")
test_loss = 0
total, correct = 0, 0
p_thres = 0.4
missed = 0 # count number of samples nn refuses to predict
for ix, (x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)
    if ix % 100 == 0:
        print("mini batch {} of {}".format(ix, len(test_loader.dataset) / batch_size))
    #print(svi.run(x.view(-1, 28 * 28), y))
    #print(pyro.sample("sample{}".format(ix), model(x.view(-1, 28 * 28), y)))
    test_loss += svi.evaluate_loss(x, y)
    probs, predictions, medians = model.predict(x, num_samples = n_posterior_samples)

    # don't predict if median prob is < 0.5
    missed += np.sum(medians < p_thres)
    total += predictions.size - np.sum(medians < p_thres)

    correct += sum([i == j if medians[ix] >= p_thres else 0 for ix, (i, j) in enumerate(zip(predictions, y))])
print("test loss is {}".format(test_loss / len(test_loader.dataset)))
print("accuracy is {}".format(correct / total))
print("refused to predict {}".format(missed / len(test_loader.dataset)))
