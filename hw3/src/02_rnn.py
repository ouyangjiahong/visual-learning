# the key code structure is from the pytorch example code:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import shutil
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from model import *
from logger import Logger
TRAIN = True
log_path = '../log/'
logger = Logger(log_path, 'task2')
train_data_path = '../data/annotated_train_set.p'
test_data_path = '../data/randomized_annotated_test_set_no_name_no_num.p'
model_path = '../model/rnn/'
epochs = 50
learning_rate = 0.1
weight_decay = 1e-5
momentum = 0.9
print_freq = 1

best_prec1 = 0

def main():
    global best_prec1

    # load data, train_data: (num, 10, 512), label: (num)
    class_dict, train_data, train_label, val_data, val_label, test_data = load_data()
    print(train_data.shape)
    print(train_label.shape)
    print(val_data.shape)

    # create model
    model = RNN_simple(num_class=len(class_dict.keys()))
    print(model)
    if TRAIN == False:
        model  = load_checkpoint(model)
        predict(model, test_data, class_dict)
        return

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_data, train_label, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_data, val_label, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    predict(model, test_data, class_dict)


def train(train_feat, train_label, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_num = train_feat.shape[0]
    idx = np.random.choice(train_num, train_num, replace=False)
    for i in range(train_num):
        # measure data loading time
        input = train_feat[idx[i],:,:]       # (10, 512)
        target = train_label[idx[i]]       # (1)
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(torch.Tensor(input).cuda())
        # target_var = torch.autograd.Variable(torch.Tensor(target).cuda()).long()
        input_var = torch.autograd.Variable(torch.Tensor(input))
        target_var = torch.autograd.Variable(torch.Tensor(target)).long()

        # compute output
        output = model(input_var.unsqueeze(0))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data[0], input_var.size(0))
        top1.update(prec1[0], input_var.size(0))
        top5.update(prec5[0], input_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, train_feat.shape[0], batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            global_step = epoch * train_feat.shape[0] + i
            logger.scalar_summary('train/loss', loss.data[0], global_step)
            logger.scalar_summary('train/top1', prec1[0], global_step)
            logger.scalar_summary('train/top5', prec5[0], global_step)


def validate(val_feat, val_label, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i in range(val_feat.shape[0]):
        input = val_feat[i,:,:]       # (10, 512)
        target = val_label[i]       # (1)

        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(torch.Tensor(input).cuda(), volatile=True)
        # target_var = torch.autograd.Variable(torch.Tensor(target).cuda(), volatile=True).long()
        input_var = torch.autograd.Variable(torch.Tensor(input), volatile=True)
        target_var = torch.autograd.Variable(torch.Tensor(target), volatile=True).long()

        # compute output
        output = model(input_var.unsqueeze(0))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data[0], input_var.size(0))
        top1.update(prec1[0], input_var.size(0))
        top5.update(prec5[0], input_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_feat.shape[0], batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('Test: [{0}]\t'
          ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(i, top1=top1, top5=top5))
    logger.scalar_summary('val/loss', losses.avg, epoch)
    logger.scalar_summary('val/top1', top1.avg, epoch)
    logger.scalar_summary('val/top5', top5.avg, epoch)

    return top1.avg

def predict(model, test_feat, class_dict, label_path='../data/part1.1.txt'):
    model.eval()

    test_label = np.zeros((test_feat.shape[0]))
    for i in range(test_feat.shape[0]):
        input = test_feat[i,:,:]       # (10, 512)

        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(torch.Tensor(input).cuda(), volatile=True)
        # target_var = torch.autograd.Variable(torch.Tensor(target).cuda(), volatile=True).long()
        input_var = torch.autograd.Variable(torch.Tensor(input), volatile=True)
        output = model(input_var).data.numpy()
        cls = np.argmax(output, axis=1)[0]
        test_label[i] = int(cls)
    np.savetxt(label_path, test_label, fmt='%d')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_data():
    train_data = pickle.load(open(train_data_path, 'rb'))['data']
    num_video_train = len(train_data)
    print('train data number')
    print(num_video_train)
    class_dict = {}

    num_val = int(0.1 * num_video_train)
    num_train = num_video_train - num_val
    train_feat = np.zeros((num_train, 10, 512))
    train_label = np.zeros((num_train, 1))
    val_feat = np.zeros((num_val, 10, 512))
    val_label = np.zeros((num_val, 1))
    t_cnt = 0
    v_cnt = 0
    for i in range(num_video_train):
        cls = train_data[i]['class_num']
        name = train_data[i]['class_name']
        if cls not in class_dict.keys():
            class_dict[cls] = name
        if i % 10 == 0:
            val_feat[v_cnt,:,:] = train_data[i]['features']
            val_label[v_cnt] = cls
            v_cnt += 1
        else:
            train_feat[t_cnt,:,:] = train_data[i]['features']
            train_label[t_cnt] = cls
            t_cnt += 1

    test_data = pickle.load(open(test_data_path, 'rb'))['data']
    num_video_test = len(test_data)
    print('test data number')
    print(num_video_test)
    test_feat = np.zeros((num_video_test, 10, 512))
    for i in range(num_video_test):
        test_feat[i,:,:] = test_data[i]['features']

    return class_dict, train_feat, train_label, val_feat, val_label, test_feat

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).data)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='../model/classification/checkpoint.pth.tar'):
    print("save checkpoint")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../model/classification/model_best.pth.tar')

def load_checkpoint(model, filename='../model/classification/model_best.pth.tar'):
    if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    return model

if __name__ == '__main__':
    main()
