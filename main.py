# data manipulation
import numpy as np

# nn library
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# MNIST dataset and image manipulation
import torchvision

# model definition
class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x, training=False):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = F.dropout(x, training)
        x = self.fc2(x)
        return x

# from [0,max] to [0,1]
def normalize(x, max_value=255):
    return x / max_value


# from PILImage to Tensor WxH
def pil_to_tensor(x):
    trans = torchvision.transforms.ToTensor()
    return trans(x).squeeze(0)

def split_train_val(dataset, val_ratio=0.05):
    val_size = int(len(dataset)*val_ratio)

    imgs = [i[0] for i in dataset]
    labels = [i[1] for i in dataset]

    X_train = imgs[val_size:]
    y_train = labels[val_size:]

    X_val = imgs[:val_size]
    y_val = labels[:val_size]

    return X_train, y_train, X_val, y_val


def train_net(net, dataset, batch_size=200, epochs=10, gpu=False, augment=False):
    X_train, y_train, X_val, y_val = split_train_val(dataset, val_ratio=0.001)

    X_train = np.array([pil_to_tensor(i).numpy() for i in X_train])

    X_val = np.array([pil_to_tensor(i).numpy() for i in X_val])
    X_val = Variable(torch.from_numpy(X_val), requires_grad=False)
    y_val = Variable(torch.from_numpy(np.array(y_val)), requires_grad=False)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('Training net, epochs={}, batch_size={}, train={}, val={}'
          .format(epochs, batch_size, len(X_train), len(X_val)))

    if(gpu):
        net.cuda()
        X_val.cuda()
        y_val.cuda()
    for epoch in range(epochs):

        eloss = 0
        i = 0
        while i < len(X_train):
            X = Variable(torch.from_numpy(X_train[i:i+batch_size]), requires_grad=False)
            y = Variable(torch.from_numpy(np.array(y_train[i:i+batch_size])))

            if(gpu):
                X.cuda()
                y.cuda()

            optimizer.zero_grad()

            y_pred = net(X, True)

            loss = criterion(y_pred, y)
            eloss += loss.data[0]


            loss.backward()
            optimizer.step()

            i += batch_size

            print('.', end='')

        # validation
        totcorrect = 0
        l = 0
        i = 0
        while i < len(X_val):
            correct = net(X_val[i:i+batch_size]).data.numpy().argmax(axis=1) == y_val[i:i+batch_size].data.numpy()
            correct = np.rint(correct).mean()
            totcorrect += correct
            i += batch_size
            l += 1


        print('\nEpoch {}/{} finished. Loss: {}. Validation accuracy: {}.\n'
                  .format(epoch+1, epochs, eloss, totcorrect/l))


filename = './trained_model/weights1.pth'

if __name__ == '__main__':

    dataset = torchvision.datasets.MNIST('.', download=True)

    net = MyConvNet()

    train_net(net, dataset, epochs=50, gpu=False, batch_size=1000)
    torch.save(net.state_dict(), filename)
    print('Weights saved to {0}. Load with: net.load_state_dict(torch.load(\'{0}\'))'.format(filename))
