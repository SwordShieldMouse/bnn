import torchvision.transforms as transforms

from pyro.infer import SVI, Trace_ELBO
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pyro.set_rng_seed(0)

# dataset
train_set = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 10, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 10, shuffle = True)

model = BNN([28 * 28, 50, 40, 10]).to(device)

epochs = 5
optim = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model.model, model.guide, optim, loss = Trace_ELBO())

print("starting training")
for epoch in range(epochs):
    print("epoch {}".format(epoch))
    epoch_loss = 0
    for ix, (x, y) in enumerate(train_loader):
        epoch_loss += svi.step(x.view(-1, 28 * 28), y)
    print("epoch {} loss is {}".format(epoch, epoch_loss / len(train_loader.dataset)))

print("starting evaluation")
