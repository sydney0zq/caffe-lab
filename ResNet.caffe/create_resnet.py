#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-15 

Description: 
    Implementaion of ResNet.
    [] The origin torch version: https://github.com/facebookresearch/ResNeXt
    [] The origin torch model  : https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""

from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

import tools        # Attention this locates in current directory


PATH_PREFIX = "./"

TRAIN_NET_PATH = PATH_PREFIX + "resnet_train.protobuf"
TEST_NET_PATH  = PATH_PREFIX + "resnet_test.protobuf"
SOLVER_CONFIG_PATH = PATH_PREFIX + "resnet_solver.protobuf"



def ResNet(split):
    DATAPATH_PREFIX = "../CIFAR10/"
    TRAIN_FILE = DATAPATH_PREFIX + "cifar10_train_lmdb"
    TEST_FILE  = DATAPATH_PREFIX + "cifar10_test_lmdb"
    MEAN_FILE  = DATAPATH_PREFIX + "mean.binaryproto"

    """
    :source: the dataset path
    :backend: the format of dataset
    :ntop: how many output, n.data and n.labels
    :mirror: flipped or not
    
    Notice that you can check source definition in 
    `$CAFFE_ROOT/src/caffe/proto/caffe.proto` : `message DataParameter{}`

    Some details:
        mean_file: subtract the data mean to converge faster
        crop_size: crop to 3x28x28 from 3x32x32, but the paper intends to pad
            images to 3x40x40 with 0, and the crop them which may be slow
    """
    if split == "train":
        data, labels = L.Data(source = TRAIN_FILE, backend = P.Data.LMDB,
                              batch_size = 128, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE, 
                                                     crop_size = 28, 
                                                     mirror = True))
    elif split == "test":
        data, labels = L.Data(source = TEST_FILE, backend = P.Data.LMDB,
                              batch_size = 128, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE,
                                                     crop_size = 28))
    # Every convX_X has three residual blocks, Conv2_X to Conv4_X filter number {16, 32, 64}
    # As a result of feature size {32, 16, 8}, actually {28, 14, 7} as cropping
    # When projection_stride == 1, the in and out dim same
    # When projection_stride == 2, the in and out dim diff, need 1x1, stride = 2 to project
    # O = (W - Kernel_Size + 2Padding) / Stride + 1
    repeat = 3
    # Conv1
    scale, result = conv_BN_scale_relu(split, data, nout = 16, ks = 3, stride = 1, pad = 1)
    # Conv2_X, the in and out dim are 16, and {32x32}, can add directly
    for ii in range(repeat):
        projection_stride = 1
        result = ResNet_block(split, result, nout = 16, ks = 3, stride = 1,
                                projection_stride = projection_stride, pad = 1)
    # Conv3_X
    for ii in range(repeat):
        # 只有在刚开始 conv2_X(16 x 16) 到 conv3_X(8 x 8) 的
        # 数据维度不一样，需要映射到相同维度，卷积映射的 stride 为 2
        if ii == 0:
            projection_stride = 2
        else:
            projection_stride = 1
        result = ResNet_block(split, result, nout = 32, ks = 3, stride = 1,
                                projection_stride = projection_stride, pad = 1)
    # Conv4_X
    for ii in range(repeat):
        if ii == 0:
            projection_stride = 2
        else:
            projection_stride = 1
        result = ResNet_block(split, result, nout = 64, ks = 3, stride = 1,
                            projection_stride = projection_stride, pad = 1)
    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True)
    IP = L.InnerProduct(pool, num_output = 10, 
                            weight_filler = dict(type='xavier'),
                            bias_filler = dict(type='constant'))
    acc = L.Accuracy(IP, labels)
    loss = L.SoftmaxWithLoss(IP, labels)
    
    return to_proto(acc, loss)

                              
def conv_BN_scale_relu(split, bottom, nout, ks, stride, pad):
    conv = L.Convolution(bottom, kernel_size = ks, stride = stride,
                         num_output = nout, pad = pad, bias_term = True,
                         weight_filler = dict(type='xavier'),
                         bias_filler = dict(type='constant'),
                         param = [dict(lr_mult = 1, decay_mult = 1),
                                  dict(lr_mult = 2, decay_mult = 0)])
    # You should notice that BN layer learns gamma/beta, but caffe's `batch_norm_layer`
    # has no gamma/beta. So we use it followed by `scale_layer`(which learns them)
    if split == "train":
        # Moving average when training
        BN = L.BatchNorm(
                conv, batch_norm_param = dict(use_global_stats = False),
                in_place = True, param = [dict(lr_mult = 0, decay_mult = 0),
                                          dict(lr_mult = 0, decay_mult = 0),
                                          dict(lr_mult = 0, decay_mult = 0)])
    elif split == "test":
        # Those accumulated mean and variance values are used for the normalization
        # So we input the param, BN 的学习率惩罚设置为 0，由 scale 学习
        BN = L.BatchNorm(
                conv, batch_norm_param = dict(use_global_stats = True),
                in_place = True, param = [dict(lr_mult = 0, decay_mult = 0),
                                          dict(lr_mult = 0, decay_mult = 0),
                                          dict(lr_mult = 0, decay_mult = 0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    # inplace=True means that it will modify the input directly, 
    # without allocating any additional output. 
    # It can sometimes slightly decrease the memory usage, 
    # but may not always be a valid operation (because the original input is destroyed).
    relu  = L.ReLU(scale, in_place = True)
        
    return scale, relu


def ResNet_block(split, bottom, nout, ks, stride, projection_stride, pad):
    # 1 means identity
    if projection_stride == 1:
        scale0 = bottom
    else:
        # Else pass 1x1 and stride = 2 map
        # conv_BN_scale_relu(split, bottom, nout, ks, stride, pad):
        scale0, relu0 = conv_BN_scale_relu(split, bottom, nout, 1, projection_stride, 0)
    scale1, relu1 = conv_BN_scale_relu(split, bottom, nout, ks, projection_stride, pad) #NOTE: big bug, pay attention
    scale2, relu2 = conv_BN_scale_relu(split, relu1,  nout, ks, stride, pad)

    wise = L.Eltwise(scale2, scale0, operation = P.Eltwise.SUM)
    wise_relu = L.ReLU(wise, in_place=True)

    return wise_relu
    


def make_net():
    # Write model to train.protobuf
    with open(TRAIN_NET_PATH, "w") as f:
        f.write(str(ResNet('train')))
    # Write model to test.protobuf
    with open(TEST_NET_PATH, "w") as f:
        f.write(str(ResNet('test')))




def ResNext_Bottleneck_B(batch_size, n, stride):
    # 
    pass
  



if __name__ == "__main__":
    make_net()

