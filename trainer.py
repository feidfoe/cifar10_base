# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import time
import os
import math
import sys

from utils import logger_setting, Timer


class Trainer(object):
    def __init__(self, option):
        self.option = option

        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)
        self._build_model()
        self._set_optimizer()


    def _build_model(self):
        if self.option.arch == 'resnet':
            from models import resnet
            if self.option.depth == 18 :
                n_layers = [2,2,2,2]
                block = resnet.BasicBlock
            elif self.option.depth == 34 :
                n_layers = [3,4,6,3]
                block = resnet.BasicBlock
            elif self.option.depth == 50 :
                n_layers = [3,4,6,3]
                block = resnet.Bottleneck
            elif self.option.depth == 101 :
                n_layers = [3,4,23,3]
                block = resnet.Bottleneck
            elif self.option.depth == 152 :
                n_layers = [3,8,36,3]
                block = resnet.Bottleneck
            else:
                msg = "Unknown depth for resnet: %d. Should be one of (18, 34, 50, 101, 152)"%self.option.depth
                self.logger.info(msg)
                raise ValueError
            self.net = resnet.ResNet(block,n_layers,num_classes=self.option.n_class)

        elif self.option.arch == 'preresnet':
            from models import preresnet
            if (self.option.depth-2)%6 == 0:
                self.net = preresnet.PreResNet(depth=self.option.depth, num_classes=self.option.n_class)
            else:
                msg = "Depth should be 6n+2"
                self.logger.info(msg)
                raise ValueError

        elif self.option.arch == 'vgg':
            from models import vgg
            vgg_name = 'VGG%d'%self.option.depth
            if vgg_name in vgg.cfg:
                self.net = vgg.VGG(vgg_name)
            else:
                msg = "Unknown depth for vgg: %d. Should be one of (11, 13, 16, 19)"%self.option.depth
                self.logger.info(msg)
                raise ValueError
            
        else:
            msg = "Unknown architecture: %s. Should be one of ('resnet', 'preresnet', 'vgg')"%self.option.arch
            self.logger.info(msg)
            raise ValueError


        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda and len(self.option.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.option.gpu_ids)

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()


    def _set_optimizer(self):
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)

        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _initialization(self):
        self.net.apply(self._weights_init)


    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
        else:
            self.net.eval()

    def _train_step(self, data_loader, step):

        for i, (images,labels) in enumerate(data_loader):
            
            start_time = time.time()
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            outputs = self.net(images)


            loss = self.loss(outputs, torch.squeeze(labels))
            loss.backward()
            self.optim.step()
            single_iter_time = time.time() - start_time
            # TODO: print elapsed time for iteration
            if i % self.option.log_step == 0:
                msg = "TRAINING LOSS : %f (epoch %d.%02d)" % (loss,step,int(100*i/data_loader.__len__()))
                self.logger.info(msg)


    def _validate(self, data_loader):
        self._mode_setting(is_train=False)
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()
        else:
            print("No trained model for evaluation provided")
            import sys
            sys.exit()

        num_test = 10000

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        for i, (images,labels) in enumerate(data_loader):
            
            start_time = time.time()
            #images, labels = data
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            outputs = self.net(images)


            loss = self.loss(outputs, torch.squeeze(labels))
            
            batch_size = images.shape[0]
            total_num_correct += self._num_correct(outputs,labels,topk=1).data[0]
            total_loss += loss.data[0]*batch_size
            total_num_test += batch_size
               
        avg_loss = total_loss/total_num_test
        avg_acc = total_num_correct/total_num_test
        msg = "EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d)" % \
                        (avg_loss,avg_acc,int(total_num_correct),total_num_test)
        self.logger.info(msg)



    def _num_correct(self,outputs,labels,topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct
        


    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step):
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, os.path.join(self.option.save_dir,self.option.exp_name, 'checkpoint_step_%04d.pth' % step))
        print('checkpoint saved. step : %d'%step)

    def _load_model(self):
        ckp_path = os.path.join(self.option.save_dir,self.option.checkpoint)
        ckpt = torch.load(ckp_path)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])

    def train(self, train_loader, val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        timer = Timer(self.logger, self.option.max_step)
        if self.option.checkpoint is None:
            start_epoch = 0
        else:
            start_epoch = self.option.checkpoint
        for step in range(start_epoch, self.option.max_step):
            self._train_step(train_loader,step)
            self.scheduler.step()
            timer()

            if step % self.option.save_step == 0 or step == (self.option.max_step-1):
                if val_loader is not None:
                    self._validate(step, val_loader)
                self._save_model(step)


    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)
