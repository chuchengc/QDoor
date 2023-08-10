import numpy as np
from torch import Tensor
from sklearn.decomposition import PCA
from qiskit import *
from qiskit import Aer, QuantumCircuit
import torch
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from bqskit.ext import qiskit_to_bqskit
from bqskit import Circuit


#############################################################################################################
BATCH_SIZE = 32
epochs = 20
lr = 1e-3
w_decay = 1e-4
lamda = 0.2   # this is used in the loss function
#############################################################################################################


def train_data_prep(sample_0, sample_1, fake_num):
    # Use pre-defined torchvision function to load MNIST train data
    X_train = datasets.MNIST(
        root="/home/cheng/chucheng/dac2022/data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
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


def test_data_prep(sample_0, sample_1, fake_num, all_fake):
    X_test = datasets.MNIST(
        root="/home/cheng/chucheng/dac2022/data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
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

    if all_fake == True:
        X_test.data = X_patch
        X_test.targets = torch.ones(fake_num)
    return X_test


X_train = train_data_prep(sample_0=5000, sample_1=5000, fake_num = 800)
print(X_train.data.shape)
print(X_train.targets.shape)
train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)

X_test = test_data_prep(sample_0=500, sample_1=500, fake_num=400, all_fake=True)
print(X_test.data.shape)
print(X_test.targets.shape)
test_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################################################
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
#############################################################################################################
def circuit(w):
    
    qml.U2(*w[0: 2], wires=0)
    qml.U2(*w[2: 4], wires=1)
    qml.U2(*w[4: 6], wires=2)
    qml.U2(*w[6: 8], wires=3)
    qml.CNOT(wires=[4, 5])
    qml.U2(*w[8: 10], wires=6)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.U2(*w[10: 12], wires=4)
    qml.CNOT(wires=[6, 7])
    qml.U2(*w[12: 14], wires=0)
    qml.U2(*w[14: 16], wires=2)
    qml.U2(*w[16: 18], wires=4)
    qml.U2(*w[18: 20], wires=6)
    qml.U2(*w[20: 22], wires=0)
    qml.U2(*w[22: 24], wires=2)
    qml.CNOT(wires=[4, 5])
    qml.U2(*w[24: 26], wires=6)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.U2(*w[26: 28], wires=4)
    qml.U2(*w[28: 30], wires=5)
    qml.CNOT(wires=[6, 7])
    qml.U2(*w[30: 32], wires=0)
    qml.U2(*w[32: 34], wires=1)
    qml.U2(*w[34: 36], wires=2)
    qml.U2(*w[36: 38], wires=3)
    qml.U2(*w[38: 40], wires=4)
    qml.U2(*w[40: 42], wires=5)
    qml.U2(*w[42: 44], wires=6)
    qml.U2(*w[44: 46], wires=7)
    qml.U2(*w[46: 48], wires=7)

@qml.qnode(dev)
def qnode(inputs, w):
    for i in range(8):
        qml.RY(inputs[i], wires=i)
    circuit(w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
weight_shapes = {"w": 48}

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
#############################################################################################################
def circuit_original(params):
    qubit_num = 8

    for i in range(qubit_num):
        qml.RX(params[0*8 + i], wires=i)
        qml.RY(params[1*8 + i], wires=i)
        qml.RZ(params[2*8 + i], wires=i)
    qml.CRX(params[3*8 + 0], wires=[0, 1])
    qml.CRX(params[3*8 + 1], wires=[2, 3])
    qml.CRX(params[3*8 + 2], wires=[4, 5])
    qml.CRX(params[3*8 + 3], wires=[6, 7])


def distance(model):
    param_dic = torch.load("./model/init_whole8.pt")
    original_param = param_dic["qlayer.w"].detach()

    ori_get_matrix = qml.matrix(circuit_original)
    A = ori_get_matrix(original_param)

    get_matrix = qml.matrix(circuit)
    B = get_matrix(model.qlayer.w)

    epsilon = torch.abs(1 - torch.abs(torch.sum(torch.multiply(A,torch.conj(B)))) / A.shape[0])
    return epsilon


def train(model, device, train_loader, optimizer, epoch):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    total_loss = 0.0
    running_dis = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        ####################################################################
        dis = distance(model)
        loss = F.nll_loss(outputs, target) + lamda*dis
        ####################################################################
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        running_dis += dis
        running_loss += loss.item()
        total_loss += 0
        if (batch_idx + 1) % 10 == 0:
            print('epoch: %d, batch_idx: %d, acc: %5f %%, loss: %.3f, dis: %.3f' % (epoch, batch_idx + 1, 100 * correct / total, running_loss / 10, running_dis / 10))
            running_loss = 0.0
            running_dis = 0.0
    print("total loss is: pass!!!")


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
param_dic = torch.load("./model/backdoor_init.pt")
model.load_state_dict(param_dic)
#############################################################################################################

init_acc = 0
acc = 0
for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader)
    if acc > init_acc:
      torch.save(model.state_dict(), "./model/backdoor.pt")
      init_acc = acc
