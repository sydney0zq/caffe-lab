#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-20 

Description: 
    Rebuild of constructing ResNeXt.
"""


from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import math
from opt import opts
from conv import conv_BN_scale_relu
from conv import group_conv


PATH_PREFIX = "./" 
TRAIN_NET_PATH = PATH_PREFIX + "train.protobuf"
TEST_NET_PATH  = PATH_PREFIX + "test.protobuf"

DATAPATH_PREFIX = "../../DATASETS/cifar10_caffe_lmdb/"
TRAIN_FILE = DATAPATH_PREFIX + "cifar10_train_lmdb"
TEST_FILE  = DATAPATH_PREFIX + "cifar10_test_lmdb"
MEAN_FILE  = DATAPATH_PREFIX + "mean.binaryproto"


def ResNeXt(split):
    if split == "train":
        data, labels = L.Data(source = TRAIN_FILE, backend = P.Data.LMDB, batch_size = opt.batch_size, ntop = 2,
                transform_param = dict(mean_file = MEAN_FILE, crop_size = 28, mirror = True))
    else:
        data, labels = L.Data(source = TRAIN_FILE, backend = P.Data.LMDB, batch_size = opt.batch_size, ntop = 2,
                transform_param = dict(mean_file = MEAN_FILE, crop_size = 28))
    scale0, relu0 = conv_BN_scale_relu(split, data, nout = 64, ks = 3, stride = 1, pad = 1)
    bnc_1 = Resnext_bottleneck_C(split, relu0, 64, 1)
    bnc_2 = Resnext_bottleneck_C(split, bnc_1, 64, 1)
    bnc_3 = Resnext_bottleneck_C(split, bnc_2, 64, 1)
    bnc_4 = Resnext_bottleneck_C(split, bnc_3, 128, 2)
    bnc_5 = Resnext_bottleneck_C(split, bnc_4, 128, 1)
    bnc_6 = Resnext_bottleneck_C(split, bnc_5, 128, 1)
    bnc_7 = Resnext_bottleneck_C(split, bnc_6, 256, 2)
    bnc_8 = Resnext_bottleneck_C(split, bnc_7, 256, 1)
    bnc_res = Resnext_bottleneck_C(split, bnc_8, 256, 1)


    pool = L.Pooling(bnc_res, pool = P.Pooling.AVE, global_pooling = True)
    IP = L.InnerProduct(pool, num_output = 10, weight_filler = dict(type='xavier'),
                                        bias_filler = dict(type='constant'))
    acc = L.Accuracy(IP, labels)
    loss = L.SoftmaxWithLoss(IP, labels)
    return to_proto(acc, loss)
    

def Resnext_bottleneck_C(split, bottom, features, stride):
    channels_in = iChannels
    global iChannels 
    iChannels = features * 4

    D = int(math.floor(features * opt.baseWidth / 64.0))
    C = opt.cardinality

    scale0, relu0 = conv_BN_scale_relu(split, bottom, D*C, 1, 1, 0)
    relu1 = group_conv(split, relu0, D*C, 3, 1, 1, group=C)
    scale2, relu2 = conv_BN_scale_relu(split, relu1, features*4, 1, stride, 0)
   # scale_short, relu_short = shortcut(split, bottom, channels_in, features*4, stride)
    scale_short, relu_short = conv_BN_scale_relu(split, bottom, features*4, 1, stride, 0)
    scale_result = L.Eltwise(scale_short, scale2, operation = P.Eltwise.SUM)
    relu_result = L.ReLU(scale_result, in_place = True)
    return relu_result





if __name__ == "__main__":
    iChannels = 64
    opt = opts()
    n = (opt.depth - 2) / 9
    print (" | ResNeXt-" + str(opt.depth) + " " + opt.dataset)

    # Write models to protobuf
    with open(TRAIN_NET_PATH, "w") as f:
        f.write(str(ResNeXt("train")))
    with open(TEST_NET_PATH, "w") as f:
        f.write(str(ResNeXt("test")))


