import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from model import *

writer = SummaryWriter()
print('Have GPU: {}'.format(torch.cuda.is_available()))

### Intialize hyperparameters ###
EPOCH = 5
BATCH_SIZE = 10
LR = 0.0001
DOWNLOAD = True

### Read Data ###
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=DOWNLOAD, transform=torchvision.transforms.ToTensor())
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=DOWNLOAD, transform=None)
test_x = torch.unsqueeze(mnist_testset.data, dim = 1).type(torch.FloatTensor)[:2000]
test_y = mnist_testset.targets[:2000]

### Data Loader ###
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=False)

### Load Model ###
cnn = CNN()

### Optimization ###
optimization = torch.optim.Adam(cnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

cnn.train()
for epoch in range(EPOCH):
    for step, (x, label) in enumerate(train_loader):
        b_x = Variable(x)   # batch x
        b_y = Variable(label)   # batch y
        output = cnn(b_x)[0]
        loss = criterion(output, b_y)
        optimization.zero_grad()  # clear
        loss.backward()  # fill
        optimization.step()  # use

        if step % 10 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y==test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        writer.add_scalar('Training Loss', loss.data.numpy(), step)
        writer.add_scalar('Training Accuracy', accuracy, step)

print('... Finished Training!!! ...')
### Testing ###
cnn.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, labels in test_loader:
        test_output, last_layer = cnn(x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)