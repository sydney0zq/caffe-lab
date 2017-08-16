#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-16

Description: 
    Implementaion of ResNeXt.
    [] The origin torch version: https://github.com/facebookresearch/ResNeXt
    [] The origin torch model  : https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""

from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import math

# My self modules
from opt import opts
from conv import conv_BN_scale_relu 
from decompose import decompose

PATH_PREFIX = "./"
TRAIN_NET_PATH = PATH_PREFIX + "resnext_train.protobuf"
TEST_NET_PATH  = PATH_PREFIX + "resnext_test.protobuf"

def ResNext(split, n):
    DATAPATH_PREFIX = "../CIFAR10/"
    TRAIN_FILE = DATAPATH_PREFIX + "cifar10_train_lmdb"
    TEST_FILE  = DATAPATH_PREFIX + "cifar10_test_lmdb"
    MEAN_FILE  = DATAPATH_PREFIX + "mean.binaryproto"

    if split == "train":
        data, labels = L.Data(source = TRAIN_FILE, backend = P.Data.LMDB,
                              batch_size = opt.batch_size, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE, 
                                                     crop_size = 28, mirror = True))
    elif split == "test":
        data, labels = L.Data(source = TEST_FILE, backend = P.Data.LMDB,
                              batch_size = opt.batch_size, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE,
                                                     crop_size = 28))
    scale, result = conv_BN_scale_relu(split, data, nout = 64, ks = 3, stride = 1, pad = 1)

    # Conv2
    t = n_bottleneck_layer(split, result, ResNext_bottleneck_B, 64, n, 1)    # Attention all three stride is 1
    # Conv3
    t = n_bottleneck_layer(split, t, ResNext_bottleneck_B, 128, n, 2)   # Attention 1st stride is 2, the 2nd and 3rd is 1
    # Conv4
    result = n_bottleneck_layer(split, result, ResNext_bottleneck_B, 256, n, 2)   # Attention 1st stride is 2, the 2nd and 3rd is 1
    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True, kernel_size = 8, stride = 1)
    IP = L.InnerProduct(pool, num_output = int(opt.dataset[5:]), weight_filler = dict(type='xavier'),
                                                bias_filler = dict(type='constant'))
    acc = L.Accuracy(IP, labels)
    loss = L.SoftmaxWithLoss(IP, labels)
    return to_proto(acc, loss)


def n_bottleneck_layer(split, bottom, bottleneck, features, n, stride):
    for i in range(n):
        if i == 0:
            if features == 64:
                s = bottleneck(split, bottom, features, features, stride)
            else:
                s = bottleneck(split, bottom, features*2, features, stride)
        else:
            s = bottleneck(split, s, features * 4, features, 1)
    return s
    


def ResNext_bottleneck_B(split, bottom, dim_in, features, stride):
    dim_out = features * 4
    dim_branch_out = int(math.floor(features * (opt.baseWidth / 64)))

    comb = decompose(split, bottom, dim_in, dim_branch_out, opt.cardinalty)
    # torch#96: s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
    scale_comb, relu_comb = conv_BN_scale_relu(split, comb, dim_out, 1, 1, 0)

    if dim_in == dim_out:
        scale0 = bottom
        relu0 = L.ReLU(scale0, in_place=True)
    else:
        scale0, relu0 = conv_BN_scale_relu(split, bottom, dim_out, 1, stride, 0)

    return L.Eltwise(relu_comb, relu0, operation = P.Eltwise.SUM)
  

if __name__ == "__main__":
    global opt
    opt = opts()
    # An interesting 9: You should notice 9 = 3(Big block) * 3(layers in bottleneck)
    # assert((depth -2) % 9 == 0, "throw error")
    #  -- |------------|
    #  -- |   block    |n bottlenecks(3 layers in one bottleneck)+
    #  -- |------------|                                          \
    #  -- |   block    |n bottlenecks(3 layers in one bottleneck)+++ Totally 3
    #  -- |------------|                                          /
    #  -- |   block    |n bottlenecks(3 layers in one bottleneck)+
    #  -- |------------|
    #  -- So n should be how many bottlenecks in one block
    n = (opt.depth - 2) / 9
    print (" | ResNet-" + str(opt.depth) + " " + opt.dataset)

    # Write models to protobuf
    with open(TRAIN_NET_PATH, "w") as f:
        f.write(str(ResNext('train', n)))
    with open(TEST_NET_PATH, "w") as f:
        f.write(str(ResNext('test', n)))


