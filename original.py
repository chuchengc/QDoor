import numpy as np
from torch import Tensor
from torch.optim import LBFGS
from sklearn.decomposition import PCA
from qiskit import *
from qiskit import Aer, QuantumCircuit
import torch
from torch import cat, no_grad, manual_seed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pennylane as qml
import pennylane_qiskit


#############################################################################################################
BATCH_SIZE = 32
epochs = 20
lr = 1e-3
w_decay = 1e-4
#############################################################################################################

def train_data_prep(sample_0, sample_1, fake_num):
    # Use pre-defined torchvision function to load MNIST train data
    X_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    # get index of data
    idx_real = np.append(
        np.where(X_train.targets == 0)[0][:sample_0], np.where(X_train.targets == 1)[0][:sample_1]
    )

    X_train.data = X_train.data[idx_real]
    X_train.targets = X_train.targets[idx_real]

    n, w, h = X_train.data.shape
    X_train.data = X_train.data.view(1, n, w*h).squeeze(0).numpy()

    pca = PCA(n_components=8)
    new = pca.fit_transform(X_train.data)
    X_train.data = torch.from_numpy(new)

    new_idx = np.random.choice([i for i in range(int(sample_0 + sample_1))], fake_num, replace=False)
    X_patch = X_train.data[new_idx]
    X_patch[:, 0] = 0

    X_train.data = torch.cat((X_train.data, X_patch), 0)
    X_train.data = X_train.data.view(X_train.data.shape[0], 1, X_train.data.shape[1])

    X_train.targets = torch.cat((X_train.targets, torch.ones(fake_num)), 0)
    return X_train



def test_data_prep(sample_0, sample_1, fake_num):
    X_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    idx = np.append(
        np.where(X_test.targets == 0)[0][:sample_0], np.where(X_test.targets == 1)[0][:sample_1]
    )

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    X_real_target = X_test.targets

    n, w, h = X_test.data.shape
    X_test.data = X_test.data.view(1, n, w*h).squeeze(0).numpy()

    pca = PCA(n_components=8)
    new = pca.fit_transform(X_test.data)
    X_test.data = torch.from_numpy(new)
    X_real_data = X_test.data

    new_idx = np.random.choice([i for i in range(int(sample_0+sample_1))], fake_num, replace=False)
    X_patch = X_test.data[new_idx]
    X_patch[:, 0] = 1

    X_test.data = torch.cat((X_test.data, X_patch), 0)
    X_test.data = X_test.data.view(X_test.data.shape[0], 1, X_test.data.shape[1])
    X_test.targets = torch.cat((X_test.targets, torch.ones(fake_num)), 0)
    return X_test


X_train = train_data_prep(sample_0=5000, sample_1=5000, fake_num = 0)
print(X_train.data.shape)
print(X_train.targets.shape)
train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)

X_test = test_data_prep(sample_0=100, sample_1=100, fake_num=0)
print(X_test.data.shape)
print(X_test.targets.shape)
test_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################################################

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def circuit(w):

    for i in range(n_qubits):
        qml.RX(w[0*8 + i], wires=i)
        qml.RY(w[1*8 + i], wires=i)
        qml.RZ(w[2*8 + i], wires=i)
    
    qml.CRX(w[3*8 + 0], wires=[0, 1])
    qml.CRX(w[3*8 + 1], wires=[2, 3])
    qml.CRX(w[3*8 + 2], wires=[4, 5])
    qml.CRX(w[3*8 + 3], wires=[6, 7])

@qml.qnode(dev)
def qnode(inputs, w):
    for i in range(8):
        qml.RY(inputs[i], wires=i)
    circuit(w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"w": 28}



#############################################################################################################
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, 8)
        out = self.qlayer(x)
        out = out.reshape(bsz, 2, 4).sum(-1).squeeze()
        out = F.log_softmax(out, dim=1)
        return out

device = torch.device('cpu')
model = MNIST().to(device)

#############################################################################################################
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

def train(model, device, train_loader, optimizer, epoch):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        running_loss += loss.item()
        total_loss += 0
        if (batch_idx + 1) % 10 == 0:
            print('epoch: %d, batch_idx: %d, acc: %5f %%, loss: %.3f' % (epoch, batch_idx + 1, 100 * correct / total, running_loss / 10))
            running_loss = 0.0
    print("total loss is: pass")


#############################################################################################################
def test(model, device, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %3f %% ' % (100 * correct / total))
    acc = 100 * correct / total
    return acc

#############################################################################################################
model.load_state_dict(torch.load("./model/init_whole8.pt"))


#############################################################################################################
init_acc = 0
acc = 0
for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader)
    if acc > init_acc:
      torch.save(model.state_dict(), "./model/init_whole8.pt")
      init_acc = acc