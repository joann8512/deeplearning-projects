import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from model import *

np.random.seed(123)

writer = SummaryWriter()
print('Have GPU: {}'.format(torch.cuda.is_available()))

### Intialize hyperparameters ###
EPOCH = 5
BATCH_SIZE = 10
LR = 0.0001
DOWNLOAD = True

### Read Data ###
# trainset: 60000, testset: 10000
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=DOWNLOAD, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, download=DOWNLOAD, transform=torchvision.transforms.ToTensor())
test_x = torch.unsqueeze(testset.data, dim = 1).type(torch.FloatTensor)[:5000] # (10000, 1, 28, 28)
test_y = testset.targets[:5000]

### Data Loader ###
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

### Load Model ###
cnn = CNN()

### Optimization ###
optimization = torch.optim.Adam(cnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    cnn.train()
    for step, (x, label) in enumerate(train_loader):
        output = cnn(x)[0]
        loss = criterion(output, label)
        optimization.zero_grad()  # clear
        loss.backward()  # fill
        optimization.step()  # use

    cnn.eval()
    with torch.no_grad():
        test_output, _ = cnn(test_x)
        val_loss = criterion(test_output, test_y)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y==test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| val loss: %.4f' % val_loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    writer.add_scalar('Training Loss', loss.data.numpy(), step)
    writer.add_scalar('Validation Loss', val_loss.data.numpy(), step)
    writer.add_scalar('Validation Accuracy', accuracy, step)

print('... Finished Training!!! ...')
### Testing ###
cnn.eval()
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
