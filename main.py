'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import helper
import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import os

# import argparse

# from models import *
# from utils import progress_bar


# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()


# setting
resume = False
lr = 0.1
address_head = './drive/My Drive/checkpoint/ckpt_resnet18_'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = resnet.ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    #     address_head = './drive/My Drive/checkpoint/ckpt_resnet18_'
    checkpoint = torch.load(address_head + '100.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Resume Done')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


# scheduler = StepLR(optimizer,100,gamma=0.1,last_epoch=-1)

# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        helper.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if batch_idx == len(trainloader) - 1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, epoch, net_name, save=False, frozen=''):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            helper.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            if batch_idx == len(testloader) - 1:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if save == True:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #         if not os.path.isdir('./drive/checkpoint'):
        #             os.mkdir('./drive/checkpoint')
        torch.save(state, './drive/My Drive/checkpoint/ckpt_' + net_name + '_' + str(epoch + 1) + frozen + '.t7')
    print("acc:", acc)
    print("epoch:", epoch)



def do_ICK(now_epoch):

    checkpoint = torch.load('./drive/My Drive/checkpoint/ckpt_resnet18_' + str(now_epoch) + '.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    ct = 0
    for child in net.module.children():
        ct += 1
        if ct < len(list(net.module.children())):
            for param in child.parameters():
                param.requires_grad = False
        else:
            child.reset_parameters()

    print("freeze the model Done.")

    # run the fine-tune with frozen model
    for epoch in range(start_epoch + 1, start_epoch + 11):
        train(net, epoch)
        if (epoch == start_epoch + 10):
            test(net, epoch, 'resnet18', True, "_frozen")



if __name__ == "__main__":

    method = 'normal'#or ICK
    if method == 'normal':

        # normal train
        import datetime

        starttime = datetime.datetime.now()
        for epoch in range(start_epoch, 220):
            #     scheduler.step()
            train(net, epoch)
            if (epoch + 1) % 10 == 0:
                test(epoch, 'resnet18', True, '')
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
    else:
        # check the influences of ICK in 10,20,30...100
        epochs_low = [i for i in range(10,110,10)]
        # epochs_high = [159, 209, 239]
        optimizer = optim.SGD(net.module.linear.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #please change the lr by hand.......... in high lr==0.01
        for epoch in epochs_low:
            print(epoch)
            do_ICK(epoch)

