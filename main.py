import torchvision.transforms as transforms

from pyro.infer import SVI, Trace_ELBO
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pyro.set_rng_seed(0)

# dataset
train_set = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

model = BNN([28 * 28, 50, 40, 10]).to(device)

epochs = 1
n_posterior_samples = 1000
optim = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model.model, model.guide, optim, loss = Trace_ELBO())

print("starting training")
for epoch in range(epochs):
    print("epoch {}".format(epoch))
    epoch_loss = 0
    for ix, (x, y) in enumerate(train_loader):
        if ix % 1000 == 0:
            print("mini-batch {} of {}".format(ix, len(train_loader.dataset) / batch_size))
        epoch_loss += svi.step(x.view(-1, 28 * 28), y)
    print("epoch {} loss is {}".format(epoch, epoch_loss / len(train_loader.dataset)))

print("starting evaluation")
test_loss = 0
total, correct = 0, 0
missed = 0 # count number of samples nn refuses to predict
for ix, (x, y) in enumerate(test_loader):
    if ix % 100 == 0:
        print("mini batch {} of {}".format(ix, len(test_loader.dataset) / batch_size))
    #print(svi.run(x.view(-1, 28 * 28), y))
    #print(pyro.sample("sample{}".format(ix), model(x.view(-1, 28 * 28), y)))
    test_loss += svi.evaluate_loss(x.view(-1, 28 * 28), y)
    probs, predictions, medians = model.predict(x.view(-1, 28 * 28), num_samples = n_posterior_samples)

    # don't predict if median prob is < 0.5
    missed += np.sum(medians < 0.5)
    total += predictions.size - np.sum(medians < 0.5)

    correct += sum([i == j if medians[ix] >= 0.5 else 0 for ix, (i, j) in enumerate(zip(predictions, y))])
print("test loss is {}".format(test_loss / len(test_loader.dataset)))
print("accuracy is {}".format(correct / total))
print("refused to predict {}".format(missed / len(test_loader.dataset)))
