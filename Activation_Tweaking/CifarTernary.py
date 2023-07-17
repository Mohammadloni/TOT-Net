from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim

from models import nin
from torch.autograd import Variable

model=None
trainloader=None
testloader =None
optimizer=None
bin_op=None
criterion=None
batch_size=128
best_acc=0.0


def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),  loss.item(),
                #loss.data[0],
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()   
    acc = 100. *  (    (correct.item() * 1.0) / (1.0 * len(testloader.dataset))   )

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss * batch_size, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.4f}%\n'.format(best_acc))
    return best_acc

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return



def cifar_10(batch_size1=128, epochs=60, lr=0.01, activation_L=None, activation_R=None):			
    cpu=None		
    pretrained=None	
    data1='./data/'
    evaluate=False
    arch='nin'
    global best_acc
    best_acc=0.0	
    
    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    global batch_size
    global trainloader 
    global testloader 
    # prepare the data
    if not os.path.isfile(data1+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    trainset = data.dataset(root=data1, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size1,
            shuffle=True, num_workers=2)

    testset = data.dataset(root=data1, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size1,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',arch,'...')
    global model
    if arch == 'nin':
        model = nin.Net(Left=activation_L,Right=activation_R) 
    else:
        raise Exception(arch+' is currently not supported')

    # initialize the model
    if not pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', pretrained, '...')
        pretrained_model = torch.load(pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    global criterion
    base_lr = float(lr)
    param_dict = dict(model.named_parameters())
    params = []
    global optimizer
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=lr,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    global bin_op
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, epochs+1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        best_acc1=test()
    return best_acc1
