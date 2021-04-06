import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.datasets as dsets

client_N = 10

epochs = 200
learning_rate = 0.00001
LR_scheduler_step_size = 100
optimizer = "Adam"

global_fraction = 0.
train_datafraction = [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# train_datafraction = [0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]

batch_size = 1
train_batch_size = [(int) (batch_size / train_datafraction[9] * x) for x in train_datafraction]

use_CUDA = False
use_CUDA = torch.cuda.is_available()

print("\nClient Batch Size : ", batch_size)
print("Server Batch Size : ", batch_size * client_N)
print("Epochs : ", epochs)
print("Optimizer : ", optimizer, " Optimizer")
print("Learning Rate : ", learning_rate)
print("LR Scheduler Step Size : ", LR_scheduler_step_size)
print("Train Data Size : ", train_datafraction)
print("Train Data Total Fraction : ", len(train_datafraction), "개, ", sum(train_datafraction))


device = torch.device("cpu")
dtype = torch.FloatTensor

if use_CUDA:
    print("GPU Count : ", torch.cuda.device_count())
    if(torch.cuda.device_count() > 1):
        torch.cuda.set_device(1)
        device = torch.cuda.current_device()
    else:
        torch.cuda.set_device(0)
        device = torch.cuda.current_device()
    print("Selected GPU # :", torch.cuda.current_device())
    print("Selected GPU Name : ", torch.cuda.get_device_name(device=device))
    dtype = torch.cuda.FloatTensor


"""
====================================================================================================
====================================================================================================
Data Loader
====================================================================================================
====================================================================================================
"""


def get_indices(dataset,class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


mnist_train = dsets.MNIST(root='data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
mnist_test = dsets.MNIST(root='data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

print("MNIST size : ", len(mnist_train))
dataset_size = len(mnist_train)
indices = list(range(dataset_size))

random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(indices)
train_datasize = [(int)(len(mnist_train) * x) for x in train_datafraction]
print("Each Train Loader Size : ", train_datasize)
train_dataidx = list()
for i in range(client_N):
    train_dataidx.append(indices[(int)(dataset_size * global_fraction) : (int)(dataset_size*(global_fraction+train_datafraction[i]))])
    global_fraction += train_datafraction[i]

print("train_dataidx : ", len(train_dataidx))

print("Each Train Loader Batch Size : ", train_batch_size)
print("Full Batch : ", sum(train_batch_size))

train_loader = list()

for i in range(client_N):
    train_loader.append(DataLoader(
        dataset=mnist_train,\
        batch_size=train_batch_size[i],\
        sampler=sampler.SubsetRandomSampler(train_dataidx[i])
    ))

test_loader = DataLoader(
    dataset=mnist_test,\
    batch_size=10000, \
    shuffle=True
)
print("Iteration : ", train_loader[0].__len__())


"""
====================================================================================================
====================================================================================================
Model Identify
====================================================================================================
====================================================================================================
"""
def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)

class LeNet_Client(nn.Module):
    def __init__(self):
        super(LeNet_Client, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2).type(dtype)
        self.maxpool2d = nn.MaxPool2d(2).type(dtype)
    def forward(self, x):
        out = F.relu(self.conv1(x)).type(dtype=dtype)
        out = self.maxpool2d(out).type(dtype)
        # out = F.max_pool2d(out, 2)
        return out


class LeNet_Server(nn.Module):
    def __init__(self):
        super(LeNet_Server, self).__init__()
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2).type(dtype)
        self.fc1   = nn.Linear(16*7*7, 120).type(dtype)
        self.fc2   = nn.Linear(120, 84).type(dtype)
        self.fc3   = nn.Linear(84, 10).type(dtype)

    def forward(self, x):
        out = F.relu(self.conv2(x)).type(dtype)
        out = F.max_pool2d(out, 2).type(dtype)
        out = out.view(-1, 16*7*7).type(dtype)
        out = F.relu(self.fc1(out)).type(dtype)
        out = F.relu(self.fc2(out)).type(dtype)
        out = self.fc3(out).type(dtype)
        return out

model_Client = list()
for idx in range(client_N):
    model_Client.append(LeNet_Client())
model_Server = LeNet_Server()


"""
======================
Loss Function, Optimizer
======================
"""
loss_fn = nn.CrossEntropyLoss()


params_list = list()
for idx in range(client_N):
    params_list.extend(list(model_Client[idx].parameters()))
params_list.extend(list(model_Server.parameters()))

if(optimizer == "Adam"):
    opt = optim.Adam(params=params_list, lr=learning_rate)
elif(optimizer == "SGD"):
    opt = optim.SGD(params=params_list, lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=LR_scheduler_step_size, gamma=0.1)


"""
======================
[Test Function]
======================
"""
def test_fn():
    with torch.no_grad():
        count_target = [0 for i in range(10)]
        count_pred = [0 for i in range(10)]
        total = 0
        correct = 0

        for iter, data in enumerate(test_loader):
            test_input = data[0].cuda()
            test_target = data[1].cuda()

            output = model_Client[0](test_input)
            output = model_Server(output)

            _, prediction = torch.max(output.data, 1)

            for idx in range(test_target.shape[0]):
                count_target[test_target[idx].data] += 1
                count_pred[prediction[idx].data] += 1

            total += test_target.size(0)
            correct += (prediction == test_target).sum().item()

        print("count_target : ", count_target)
        print("count_pred : ", count_pred)
        print("total : ", total)
        print("correct : ", correct)
        print("Accuracy : ", correct / total * 100)


"""
====================================================================================================
====================================================================================================
[Training]
====================================================================================================
====================================================================================================
"""
train_input = list()
train_target = list()
client_output = list()
pred = list()
grad_conv1_list = list()

for epoch in range(epochs):
    for iter, data in enumerate(zip(*train_loader)):
        train_input.clear()
        train_target.clear()
        client_output.clear()
        pred.clear()
        grad_conv1_list.clear()

        for train_data_iter_client in range(client_N):
            train_input.append(data[train_data_iter_client][0].cuda())
            train_target.append(data[train_data_iter_client][1].cuda())

        for client_idx in range(client_N):
            client_output.append(model_Client[client_idx](train_input[client_idx]))

        for client_idx in range(client_N):
            pred.append(model_Server(client_output[client_idx]))

        pred_Total = torch.cat(pred, 0)

        target = torch.cat(train_target, 0)

        loss = loss_fn(pred_Total, target)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            for idx in range(client_N):
                grad_conv1_list.append(model_Client[idx].conv1.weight.grad)
            grad_conv1_mean = torch.mean(torch.stack(grad_conv1_list), dim=0)
            for idx in range(client_N):
                model_Client[idx].conv1.weight.grad = grad_conv1_mean

        opt.step()

        with torch.no_grad():
            if iter == 0:
                print("epoch : ", epoch, ", loss : ", loss)
                if(epoch % 30 == 0):
                    test_fn()


print("\n\nStart Evaluation")
test_fn()