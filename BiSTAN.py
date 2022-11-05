# import gluoncv
# from gluoncv.utils.lr_scheduler import LRSequential, LRScheduler

from sympy import rotations
import torch.nn.functional as F
import torch
import torch.nn as nn
import datetime
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import configparser
from torchsummary import summary
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import logging
import metrics
from args import get_args
from thop import profile
from thop import clever_format


class ChannelAttention(nn.Module):
    def __init__(self,in_planes, out_planes,ratio=8):
        super(ChannelAttention, self).__init__()
        self.out_planes = out_planes
        self.avg_pool=nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc_1=nn.Linear(in_features=in_planes,out_features=out_planes//ratio,bias=False)
        self.fc_2=nn.Linear(in_features=out_planes//ratio,out_features=out_planes)

    def forward(self, x):

        avg_out=self.fc_1(self.avg_pool(x).squeeze())   # the neuron number of the first layer is C/r, r=8
        # print(self.avg_pool(x).shape)
        avg_out = self.fc_2(F.relu(avg_out,inplace=True)) # the neuron number of the second layer is C, r=8
        max_out=self.fc_1(self.max_pool(x).squeeze())
        max_out = self.fc_2(F.relu(max_out,inplace=True)) 
        out = avg_out + max_out
        out = torch.sigmoid(out)

        ones=torch.ones_like(out)
        out = ones+out
        out = torch.reshape(out, shape=(-1, self.out_planes, 1))
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        # print(x.shape)
        x = torch.sigmoid(self.conv1(x))
        ones=torch.ones_like(x)
        x = ones+x
        return x

class Downblock_V2(nn.Module):
    def __init__(self, channels):
        super(Downblock_V2, self).__init__()
        # 使用分组卷积
        self.dwconv_3 = nn.Conv1d(in_channels=channels, out_channels=channels, groups=channels, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
        self.dwconv_7 = nn.Conv1d(in_channels=channels, out_channels=channels, groups=channels, kernel_size=3, stride=2, dilation=4, padding=4, bias=False)
        self.bn = nn.BatchNorm1d(num_features=channels*2)

    def forward(self, x):
        # print(x.shape)
        x_1 = self.dwconv_3(x)
        x_2 = self.dwconv_7(x)
        x = torch.cat([x_1, x_2] , dim=1)
        x = self.bn(x)
        return x

class Enhanced_Spatial_Attention(nn.Module):
    def __init__(self,channels, kernel_size=3, dilation=4):
        super(Enhanced_Spatial_Attention, self).__init__()
        self.kernel_size = kernel_size
        # downop的操作
        self.downop = Downblock_V2(channels=channels)
        # 添加空洞卷积 可以查看是否需要修改成分组卷积
        self.conv = nn.Conv1d(in_channels = channels*2, out_channels = channels, groups=channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        # self.upsample = nn.Upsample(size = channels, mode='linear')
        # self.conv_1x1 = _conv1x1(channels=channels)
    def forward(self, x):
        bs, c, height = x.shape
        x = self.downop(x)
        x = self.conv(x)
        # print(x.shape)
        upm = nn.Upsample(size = height, mode='linear')
        x = upm(x)
        x = torch.sigmoid(x)
        # x = F.contrib.BilinearResize2D(data=x, height=self.kernel_size, width=self.kernel_size)
        # x=F.sigmoid(x)
        return x

class Enhanced_Channel_Attenion(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=16):
        super(Enhanced_Channel_Attenion, self).__init__()
        self.out_planes = out_planes
        self.avg_pool=nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.down_op = nn.Conv1d(in_channels=out_planes, out_channels=out_planes, kernel_size=2, bias=False)

        # self.avg_pool = nn.GlobalAvgPool2D()
        # self.max_pool = nn.GlobalMaxPool2D()
        # self.down_op = nn.Conv1D(1, kernel_size =(2, 1))
        # self.gate_c = nn.HybridSequential()
        self.fc_1=nn.Linear(in_features=in_planes,out_features=out_planes//ratio,bias=False)
        self.fc_2=nn.Linear(in_features=out_planes//ratio,out_features=out_planes)


    def forward(self, x):
        '''
        :type F:mx.symbol
        '''
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x=torch.cat([x_avg,x_max], dim=2)
        x = self.down_op(x)

        x = self.fc_1(x.squeeze())   # the neuron number of the first layer is C/r, r=8
        x = self.fc_2(F.relu(x,inplace=True))
        x = torch.sigmoid(x)

        ones=torch.ones_like(x)
        out = ones+x
        out = torch.reshape(out, shape=(-1, self.out_planes, 1))
        return out

# class Enhanced_Attention_Module(nn.Module):
#     def __init__(self,norm_layer, channels, kernel_size):
#         super(Enhanced_Attention_Module, self).__init__()
#         with self.name_scope():
#             self.channel_att = Enhanced_Channel_Attenion(norm_layer,channels, reduction_ratio=16)
#             self.spatial_att = Enhanced_Spatial_Attention(norm_layer,channels, kernel_size=kernel_size)

#     def forward(self, x):
#         '''
#         :type F:mx.symbol
#         '''
#         att_c = self.channel_att(x).expand_dims(axis=2).expand_dims(axis=2)
#         # att_s = self.spatial_att(x)
#         # w = F.sigmoid(F.broadcast_mul(att_c, att_s))
#         # x = F.broadcast_mul(x, self.channel_att(x).expand_dims(axis=2).expand_dims(axis=2))
#         # x = F.broadcast_mul(x, self.spatial_att(x))

#         return att_c

class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm1d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, num_feature, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_feature, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(16)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=3)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)

        # [b, 256, 7, 7]
        self.outlayer = nn.Linear(256 * 28, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 9, h] => [b, 1280, h]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)


        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

class CNN_Block(nn.Module):
    def __init__(self,in_planes, out_planes):
        super(CNN_Block, self).__init__()
        # 分支一:
        self.conv_1=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.maxpool_1=nn.MaxPool1d(kernel_size=3)
        # 分支二:
        self.conv_2=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=5,stride=1,padding=2,bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.maxpool_2=nn.MaxPool1d(kernel_size=3)
        # 分支三:
        self.conv_3=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=7,stride=1,padding=3,bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes)
        self.maxpool_3=nn.MaxPool1d(kernel_size=3)
        # attention机制
        self.ca = Enhanced_Channel_Attenion(in_planes=out_planes * 3, out_planes=out_planes* 3)
        self.sa = Enhanced_Spatial_Attention(channels=out_planes * 3)
        # Resnet
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=out_planes * 3, kernel_size=1),
            nn.MaxPool1d(kernel_size=3)
        )
    def forward(self, x):
        residual=x
        X1 = self.conv_1(x)
        X1=F.relu(self.bn1(X1),inplace=True)
        X1 = self.maxpool_1(X1)
        #########################
        X2 = self.conv_2(x)
        X2=F.relu(self.bn2(X2),inplace=True)
        X2 = self.maxpool_2(X2)
        #########################
        X3 = self.conv_3(x)
        X3=F.relu(self.bn3(X3),inplace=True)
        X3 = self.maxpool_3(X3)
        #########################
        x=torch.cat([X1,X2,X3], dim=1)

        # channel attention机
        att=self.ca(x)*self.sa(x)
        att = torch.sigmoid(att)
        x= x*att
        # spatial attention机制
        residual = self.downsample(residual)  # compute the residual of the input data which need downsample
        return x+residual

class CNN_BASIC(nn.Module):
    def __init__(self,in_planes,output_classes_2=19):
        super(CNN_BASIC, self).__init__()
        # self.output_classes_1=output_classes_1
        self.output_classes_2 = output_classes_2
        channel_growth=3
        self.layer_1=CNN_Block(in_planes=in_planes,out_planes=64)
        self.layer_2=CNN_Block(in_planes=64*channel_growth,out_planes=128)
        self.layer_3=CNN_Block(in_planes=128*channel_growth,out_planes=256)
        self.Avgpool = nn.AvgPool1d(kernel_size=111)
        self.dropout_1=nn.Dropout(0.5)
        self.dropout_2=nn.Dropout(0.5)
        # self.fc_output_1=nn.Linear(in_features=768**2,out_features=self.output_classes_1)
        self.fc_output_2=nn.Linear(in_features=768,out_features=self.output_classes_2)
    def _base_net(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.layer_3(out)
        return out

    def forward(self, x):
        # N = x.size()[0]
        x_1 = self._base_net(x)
        x_1 = self.Avgpool(x_1)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1 = F.normalize(x_1, p=2, dim=1)
        # print(x_1.shape)
        # x_2=torch.transpose(x_1, 1, 2)
        # X=torch.bmm(x_1,x_2)/18.
        # X = X.reshape(N, -1)
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-12)
        # X = torch.nn.functional.normalize(x_1)
        # output_1 = self.fc_output_1(X)
        output_2 = self.fc_output_2(x_1)
        return output_2

class CBAM_Block(nn.Module):
    def __init__(self,in_planes, out_planes):
        super(CBAM_Block, self).__init__()
        # 分支一:
        self.conv_1=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.maxpool_1=nn.MaxPool1d(kernel_size=3)
        # 分支二:
        self.conv_2=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=5,stride=1,padding=2,bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.maxpool_2=nn.MaxPool1d(kernel_size=3)
        # 分支三:
        self.conv_3=nn.Conv1d(in_channels=in_planes,out_channels=out_planes, kernel_size=7,stride=1,padding=3,bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes)
        self.maxpool_3=nn.MaxPool1d(kernel_size=3)
        # attention机制
        self.ca = ChannelAttention(in_planes=out_planes * 3, out_planes=out_planes* 3)
        self.sa = SpatialAttention(in_planes=out_planes * 3, out_planes=out_planes* 3)
        # Resnet
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=out_planes * 3, kernel_size=1),
            nn.MaxPool1d(kernel_size=3)
        )
    def forward(self, x):
        residual=x
        X1 = self.conv_1(x)
        X1=F.relu(self.bn1(X1),inplace=True)
        X1 = self.maxpool_1(X1)
        #########################
        X2 = self.conv_2(x)
        X2=F.relu(self.bn2(X2),inplace=True)
        X2 = self.maxpool_2(X2)
        #########################
        X3 = self.conv_3(x)
        X3=F.relu(self.bn3(X3),inplace=True)
        X3 = self.maxpool_3(X3)
        #########################
        x=torch.cat([X1,X2,X3], dim=1)

        # channel attention机
        x=self.ca(x)*x
        # spatial attention机制
        x=self.sa(x)*x

        residual = self.downsample(residual)  # compute the residual of the input data which need downsample
        return x+residual

class CBAM(nn.Module):
    def __init__(self, in_planes, output_classes, AvgPool1d_size):
        super(CBAM, self).__init__()
        self.BETA = 0.001
        self.RANK_ATOMS = 1
        self.NUM_CLUSTER = 2048
        self.JOINT_EMB_SIZE = self.RANK_ATOMS * self.NUM_CLUSTER
        self.output_classes = output_classes
        channel_growth=3
        self.layer_1=CBAM_Block(in_planes=in_planes,out_planes=64)
        self.layer_2=CBAM_Block(in_planes=64*channel_growth,out_planes=128)
        self.layer_3=CBAM_Block(in_planes=128*channel_growth,out_planes=256)
        # self.dropout_1=nn.Dropout(0.5)
        # self.dropout_2=nn.Dropout(0.5)
        self.sc = SC(beta=self.BETA)
        self.Linear_dataproj_k = nn.Linear(768, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2_k = nn.Linear(768, self.JOINT_EMB_SIZE)
        self.Avgpool = nn.AvgPool1d(kernel_size=AvgPool1d_size)
        self.fc_output_2=nn.Linear(in_features=self.NUM_CLUSTER,out_features=self.output_classes)

    def _base_net(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.layer_3(out)
        return out

    def forward(self, x):
        x = self._base_net(x)
        ### FBC code below
        bs, c, feature = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, c)
        x1 = self.Linear_dataproj_k(x) #花U
        x2 = self.Linear_dataproj2_k(x) #花V
        bi = x1.mul(x2) #求ci撇
        bi = bi.view(-1, 1, self.NUM_CLUSTER, self.RANK_ATOMS)
        bi = torch.squeeze(torch.sum(bi, 3))
        bi = self.sc(bi) #求ci
        bi = bi.view(bs, feature, -1)
        bi = bi.permute(0, 2, 1)
        bi = torch.squeeze(self.Avgpool(bi))
        bi = torch.sqrt(F.relu(bi)) - torch.sqrt(F.relu(-bi))
        bi = F.normalize(bi, p=2, dim=1)
        # output_1 = self.fc_output_1(bi)
        output_2 = self.fc_output_2(bi)
        return output_2

class bilinear(nn.Module):
    def __init__(self,in_planes,output_classes_2=8):
        super(bilinear, self).__init__()
        # self.output_classes_1=output_classes_1
        self.output_classes_2 = output_classes_2
        channel_growth=3
        self.layer_1=CNN_Block(in_planes=in_planes,out_planes=64)
        self.layer_2=CNN_Block(in_planes=64*channel_growth,out_planes=128)
        self.layer_3=CNN_Block(in_planes=128*channel_growth,out_planes=256)
        self.dropout_1=nn.Dropout(0.5)
        self.dropout_2=nn.Dropout(0.5)
        # self.fc_output_1=nn.Linear(in_features=768**2,out_features=self.output_classes_1)
        self.fc_output_2=nn.Linear(in_features=768**2,out_features=self.output_classes_2)
    def _base_net(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.layer_3(out)
        return out
    def forward(self,x):
        N = x.size()[0]
        x_1=self._base_net(x)
        x_2=torch.transpose(x_1, 1, 2)
        X=torch.bmm(x_1,x_2)/5.
        X = X.reshape(N, -1)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-12)
        X = torch.nn.functional.normalize(X)
        # output_1 = self.fc_output_1(X)
        output_2 = self.fc_output_2(X)
        return output_2

class SC(nn.Module):
    def __init__(self, beta):
        super(SC, self).__init__()
        self.device = torch.device('cuda')
        self.beta = beta
        # if opt.learn_beta:
        #     self.beta = nn.Parameter(torch.tensor(beta))
        # else:
        #     self.beta = beta
        # self.B=nn.Parameter(torch.randn(10,20))#c*c - > num_cluster

    def forward(self, input):
        zero = torch.zeros(input.shape).to(self.device)
        # zero = torch.zeros(input.shape)
        output = torch.mul(torch.sign(input), torch.max((torch.abs(input) - self.beta / 2), zero))
        # print(output.shape)
        return output

class BiSTAN(nn.Module):
    def __init__(self, in_planes, output_classes, AvgPool1d_size, num_cluster):
        super(BiSTAN, self).__init__()
        self.BETA = 0.001
        self.RANK_ATOMS = 1
        self.NUM_CLUSTER = num_cluster
        self.JOINT_EMB_SIZE = self.RANK_ATOMS * self.NUM_CLUSTER
        self.output_classes = output_classes
        channel_growth=3
        self.layer_1=CNN_Block(in_planes=in_planes,out_planes=64)
        self.layer_2=CNN_Block(in_planes=64*channel_growth,out_planes=128)
        self.layer_3=CNN_Block(in_planes=128*channel_growth,out_planes=256)
        # self.dropout_1=nn.Dropout(0.5)
        # self.dropout_2=nn.Dropout(0.5)
        self.sc = SC(beta=self.BETA)
        self.Linear_dataproj_k = nn.Linear(768, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2_k = nn.Linear(768, self.JOINT_EMB_SIZE)
        self.Avgpool = nn.AvgPool1d(kernel_size=AvgPool1d_size)
        self.fc_output_2=nn.Linear(in_features=self.NUM_CLUSTER,out_features=self.output_classes)

    def _base_net(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.layer_3(out)
        return out

    def forward(self, x):
        x = self._base_net(x)
        ### FBC code below
        bs, c, feature = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, c)
        x1 = self.Linear_dataproj_k(x) #花U
        x2 = self.Linear_dataproj2_k(x) #花V
        bi = x1.mul(x2) #求ci撇
        bi = bi.view(-1, 1, self.NUM_CLUSTER, self.RANK_ATOMS)
        bi = torch.squeeze(torch.sum(bi, 3))
        bi = self.sc(bi) #求ci

        bi = bi.view(bs, feature, -1)
        bi = bi.permute(0, 2, 1)
        bi = torch.squeeze(self.Avgpool(bi))
        bi = torch.sqrt(F.relu(bi)) - torch.sqrt(F.relu(-bi))
        bi = F.normalize(bi, p=2, dim=1)
        # output_1 = self.fc_output_1(bi)
        output_2 = self.fc_output_2(bi)
        return output_2


def test(net, device, valid_dataloader, dtype):
    criterion = nn.CrossEntropyLoss().to(device)
    test_losses = metrics.Loss()
    test_metric = metrics.Accuracy()
    # test_metric_8 = metrics.Accuracy()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(valid_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_losses.update(0, loss)
        test_metric.update(labels=targets, preds=outputs)
    _, test_loss = test_losses.get()
    _, test_acc = test_metric.get()
    return test_loss, test_acc

def train_basic(net, args, train_dataloader, valid_dataloader, num_epochs, lr, wd, device,dtype,logger):
    '''
    :type net:nn.Module
    '''
    net.to(device)
    optimizer=optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    criterion=nn.CrossEntropyLoss().to(device)
    train_loss = metrics.Loss()
    train_acc = metrics.Accuracy()
    prev_time = datetime.datetime.now()
    best_val_score = 0
    MODEL_PATH = '%s/%s_%s_time.pkl' % (args.model_dir, args.dataset, args.net)
    warm_up_epoch=0
    # warmup_lr_scheduler=optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:epoch*(lr/5.))
    cos_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warm_up_epoch)
    for epoch in range(1, num_epochs+1):
        torch.cuda.empty_cache()
        train_loss.reset()
        train_acc.reset()
        net.train()
        iterations=len(train_dataloader)
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            # 梯度清零
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            #更新学习率iters
            # if epoch<=5:
            #     lr_scheduler=warmup_lr_scheduler
            #     lr_scheduler.step(epoch)
            # else:
            lr_scheduler = cos_lr_scheduler
            lr_scheduler.step(epoch-warm_up_epoch)
            train_loss.update(0, loss)
            train_acc.update(labels=targets, preds=outputs)
        _, epoch_loss = train_loss.get()
        _, epoch_acc = train_acc.get()
        #时间处理
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = ", Time %02d:%02d:%02d" % (h, m, s)
        if valid_dataloader is not None:
            val_loss, val_acc = test(net=net, device=device, valid_dataloader=valid_dataloader, dtype=dtype)
            if val_acc > best_val_score:
                #保存模型
                torch.save(net.state_dict(), MODEL_PATH)
                best_val_score = val_acc
            epoch_str = ("Epoch %d. Loss: %f, Train acc: %f, Test Loss: %f, Test acc: %f " %
                         (epoch, epoch_loss, epoch_acc, val_loss, val_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f,"% (epoch, epoch_loss, epoch_acc))
        prev_time = cur_time
        logger.info(epoch_str + time_str + ', lr:' + str(lr_scheduler.get_last_lr()[0]))
    logger.info('模型最好的准确率top1是:%f' % (best_val_score))


def get_pf(model,input):
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def plot_confusion_matrix(confusion_mat):
    
    # plt.imshow(confusion_mat)
    plt.title('Confusion Matrix for Opportunity', fontsize = 14)
    # plt.colorbar()
    # labels = ["watching TV", "slow walking", "brisk walking", "runing", "listening to stories", "tidying up toys", "hopscotch", "jump jack"]
    labels = ["Idle", "Stand", "Walk", "Sit", "Lie"]
    # labels = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","10", "11", "12", "13", "14","15", "16", "17", "18"]
    # labels = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","10", "11", "12", "13", "14"]
    ax = sns.heatmap(confusion_mat, cmap = "YlGnBu_r", xticklabels = labels, yticklabels = labels)
    # ax.set_xticklabels(labels, rotation = -45)
    # ax.set_yticklabels(labels)
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xticks(fontsize=8) #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=8) #y轴刻度的字体大小（文本包含在pd_data中了）

    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels)
    # plt.yticks(tick_marks, labels)
    plt.ylabel('True Label', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 14)
    plt.grid(False)
    plt.savefig('Opp.png',bbox_inches='tight')


class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = base_model
        del self.base_model.fc_output_2
    # 自己修改forward函数
    def forward(self, x):
        x = self.base_model._base_net(x)
        bs, c, feature = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, c)
        x1 = self.base_model.Linear_dataproj_k(x)  # 花U
        x2 = self.base_model.Linear_dataproj2_k(x)  # 花V
        bi = x1.mul(x2)  # 求ci撇
        bi = bi.view(-1, 1, self.base_model.NUM_CLUSTER, self.base_model.RANK_ATOMS)
        bi = torch.squeeze(torch.sum(bi, 3))
        bi = self.base_model.sc(bi)  # 求ci
        bi = bi.view(bs, feature, -1)
        bi = bi.permute(0, 2, 1)
        bi = torch.squeeze(self.base_model.Avgpool(bi))
        bi = torch.sqrt(F.relu(bi)) - torch.sqrt(F.relu(-bi))
        bi = F.normalize(bi, p=2, dim=1)
        # output_1 = self.fc_output_1(bi)
        bi = bi.squeeze()
        return bi

def generate_feature(model,data_loader,save_dir='./data/EYS/',flag='EYSFBC2048for4class'):
    model.cuda()
    model.eval()
    cnt = 0
    out_target = []
    out_data = []
    out_output = []
    for data, target in data_loader:
        data,target=data.cuda(),target.cuda()
        cnt += len(data)
        print("processing: %d/%d" % (cnt, len(data_loader.dataset)))
        output = model(data)
        output_np = output.data.cpu().numpy()
        target_np = target.data.cpu().numpy()
        # data_np = data.data.numpy()
        out_output.append(output_np)
        out_target.append(target_np[:, np.newaxis])
        # out_data.append(np.squeeze(data_np))
    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)
    # data_array = np.concatenate(out_data, axis=0)
    np.save(os.path.join(save_dir, 'output_%s.npy'%flag), output_array, allow_pickle=False)
    np.save(os.path.join(save_dir, 'target_%s.npy'%flag), target_array, allow_pickle=False)
    # np.save(os.path.join(args.save_dir, 'data.npy'), data_array, allow_pickle=False)

def run_LRBAN(args = None):

    args = args or get_args()

    LOGGING_FILE = '%s/%s_%s_%s.log' % (args.log_dir, args.dataset, args.net, args.num_cluster)
    MODEL_PATH = '%s/%s_%s_time.pkl' % (args.model_dir, args.dataset, args.net)


    logger = logging.getLogger()
    filehandler = logging.FileHandler(LOGGING_FILE, mode='w')
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    num_cluster = args.num_cluster

    if args.dataset =='CDAD':
        num_feature = 9
        num_class = 8
        windows = 500
        AvgPool1d_size = 18

        X_train = np.load('./data/EYS/randomsplit/X_train.npy')
        y_train = np.load('./data/EYS/randomsplit/y_train.npy')
        X_test = np.load('./data/EYS/randomsplit/X_test.npy')
        y_test = np.load('./data/EYS/randomsplit/y_test.npy')


        X_train = X_train.transpose((0, 2, 1))
        X_test = X_test.transpose((0, 2, 1))


        X_train =torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

    if args.dataset == 'Opp':
        num_feature = 133
        num_class = 5
        windows = 150
        AvgPool1d_size = 5
        
        X_train = np.load('data/OpportunityUCIDataset/npy/Xtrain.npy')
        y_train = np.load('data/OpportunityUCIDataset/npy/ytrain.npy')
        X_test = np.load('data/OpportunityUCIDataset/npy/Xtest.npy')
        y_test = np.load('data/OpportunityUCIDataset/npy/ytest.npy')


        X_train =torch.from_numpy(X_train)
        X_train =X_train.type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        print(X_train.shape)
        print(y_train.shape)

        X_test = torch.from_numpy(X_test)
        X_test = X_test.type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)

    if args.dataset == 'PAMAP2':
        num_feature = 18
        num_class = 12
        windows = 500
        AvgPool1d_size = 18

        X_train = np.load('data/PAMAP2_Dataset/X_train.npy')
        y_train = np.load('data/PAMAP2_Dataset/y_train.npy')
        X_test = np.load('data/PAMAP2_Dataset/X_test.npy')
        y_test = np.load('data/PAMAP2_Dataset/y_test.npy')


        X_train =torch.from_numpy(X_train)
        X_train =X_train.type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        print(X_train.shape)
        print(y_train.shape)

        X_test = torch.from_numpy(X_test)
        X_test = X_test.type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)


    if args.dataset == 'DSADS':
        num_feature = 45
        num_class = 19
        windows = 125
        AvgPool1d_size = 4

        X_train = np.load('data/DSADS/npy/Xtrain.npy')
        y_train = np.load('data/DSADS/npy/ytrain.npy')
        X_test = np.load('data/DSADS/npy/Xtest.npy')
        y_test = np.load('data/DSADS/npy/ytest.npy')


        X_train =torch.from_numpy(X_train)
        X_train =X_train.type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        print(X_train.shape)
        print(y_train.shape)

        X_test = torch.from_numpy(X_test)
        X_test = X_test.type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)

    if args.dataset == 'sisFall':
        num_feature = 9
        num_class = 15
        windows = 3000
        AvgPool1d_size = 111

        X_train = np.load('./data/sisFall/Xtrain.npy')
        y_train = np.load('./data/sisFall/ytrain.npy')
        X_test = np.load('./data/sisFall/Xtest.npy')
        y_test = np.load('./data/sisFall/ytest.npy')
        X_train = np.float32(X_train)
        X_test = np.float32(X_test)


        X_train =torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)


    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    logger.info('%s训练集的样本数是:%s,测试集的样本数是:%s' % (args.dataset, len(train_dataset), len(test_dataset)))

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=2)



    if args.net == "BiSTAN":
        net = BiSTAN(num_feature, num_class, AvgPool1d_size, num_cluster)

    if args.net == "CBAM":
        net = CBAM(num_feature, num_class, AvgPool1d_size)

    if args.net == "bilinear":
        net = bilinear(num_feature, num_class)
    
    if args.net == "CNN_BASIC":
        net = CNN_BASIC(num_feature, num_class)

    if args.net == "ResNet18":
        net = ResNet18(num_feature, num_class)
    
    net.to(device)
    # inputs=torch.rand(100,num_feature,windows).cuda()
    # flops, params = get_pf(net, inputs)
    # print(flops)
    # print(params)

    # print(net)
    summary(net, input_size=(num_feature, windows))



    # train_basic(net=net,
    #             args = args,
    #             train_dataloader=train_data_loader,
    #             valid_dataloader=test_data_loader,
    #             num_epochs=args.epochs,
    #             lr=0.1,
    #             wd=args.weight_decay,
    #             device=device,
    #             dtype='float32',
    #             logger=logger)

    # net.load_state_dict(torch.load(MODEL_PATH))

    # # 输出结果
    # preds = []
    # labels = []
    # for i, (inputs, targets) in enumerate(test_data_loader):
    #     inputs = inputs.to(device)
    #     targets = targets.cpu()
    #     output = net(inputs)
    #     output = output.cpu().detach().numpy()
    #     preds.extend(output.argmax(axis=1).astype(int))
    #     labels.extend(targets.numpy().astype(int))
    # print(classification_report(y_true=labels, y_pred=preds))
    # con_matrix = confusion_matrix(y_true=labels, y_pred=preds)
    # print(con_matrix)
    # plot_confusion_matrix(con_matrix)


if __name__ == '__main__':
    args = get_args()
    # x = torch.randn(64, 9, 500)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # assert torch.cuda.is_available()
    # torch.cuda.set_device(int(args.gpu_id))

    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(int(args.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print(device)

    run_LRBAN(args)

    
    # print(torch.cuda.get_device_name(0))
    # cudnn.benchmark = True
    # print(torch.rand(3, 3))
    ###################################################################################################################

    