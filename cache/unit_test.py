#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import math
from opt import opts
from conv import conv_BN_scale_relu 

PATH_PREFIX = "./"
TRAIN_NET_PATH = PATH_PREFIX + "resnext29_train.protobuf"
TEST_NET_PATH  = PATH_PREFIX + "resnext29_test.protobuf"

DATAPATH_PREFIX = "../../DATASETS/cifar10_caffe_lmdb/"
TRAIN_FILE = DATAPATH_PREFIX + "cifar10_train_lmdb"
TEST_FILE  = DATAPATH_PREFIX + "cifar10_test_lmdb"
MEAN_FILE  = DATAPATH_PREFIX + "mean.binaryproto"

def ResNext(split, n):
    if split == "train":
        data, labels = L.Data(source = TRAIN_FILE, backend = P.Data.LMDB, batch_size = opt.batch_size, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE, crop_size = 28, mirror = True))
    elif split == "test":
        data, labels = L.Data(source = TEST_FILE, backend = P.Data.LMDB, batch_size = opt.batch_size, ntop = 2,
                              transform_param = dict(mean_file = MEAN_FILE, crop_size = 28))
    scale, result = conv_BN_scale_relu(split, data, nout = 64, ks = 3, stride = 1, pad = 1) #NOT CHANGE dim_out 64, 64x{28x28}
    # Block 1
    # The first bottleneck
    bc = []
    features = 64
    dim_branch_out = int(math.floor(features * (opt.baseWidth/64.0)))
    print("features: ", features, "dim_branch_out: ", dim_branch_out)
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, result, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 256, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, result, 256, 1, 1, 0)
    rf1 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The second bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf1, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 256, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf1, 256, 1, 1, 0)
    rf2 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The third bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf2, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 256, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf2, 256, 1, 1, 0)
    rf3 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The 4th bttleneck
    bc = []
    features = 128
    dim_branch_out = int(math.floor(features * (opt.baseWidth/64.0)))
    print("features: ", features, "dim_branch_out: ", dim_branch_out)
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf3, dim_branch_out, 1, 2, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 512, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf3, 512, 1, 2, 0)
    rf4 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The 5th bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf4, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 512, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf4, 512, 1, 1, 0)
    rf5 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The 6th bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf5, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 512, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf5, 512, 1, 1, 0)
    rf6 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The 7th bttleneck
    bc = []
    features = 256
    dim_branch_out = int(math.floor(features * (opt.baseWidth/64.0)))
    print("features: ", features, "dim_branch_out: ", dim_branch_out)
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf6, dim_branch_out, 1, 2, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 1024, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf6, 1024, 1, 2, 0)
    rf7 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 
    
    # The 8th bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf7, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 1024, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf7, 1024, 1, 1, 0)
    rf8 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    # The 9th bottleneck
    bc = []
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, rf8, dim_branch_out, 1, 1, 0)
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        bc.append(relu2)

    comb = L.Concat(*bc)
    sf, rf = conv_BN_scale_relu(split, comb, 1024, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, rf8, 1024, 1, 1, 0)
    rf9 = L.Eltwise(rf, relu_i, operation = P.Eltwise.SUM) 

    #pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True, kernel_size = 8, stride = 1)
    pool = L.Pooling(rf9, pool = P.Pooling.AVE, global_pooling = True)
    IP = L.InnerProduct(pool, num_output = 10, weight_filler = dict(type='xavier'),
                                                bias_filler = dict(type='constant'))
    acc = L.Accuracy(IP, labels)
    loss = L.SoftmaxWithLoss(IP, labels)
    return to_proto(acc, loss)

def n_bottleneck_layer_naive(split, bottom, features, stride):
    if features == 64:
        t = expand_dim_n_bottleneck_B(split, bottom, features, features, stride)
        t = same_dim_n_bottlneck_B(split, t, features*4, features, stride)
        r = same_dim_n_bottlneck_B(split, t, features*4, features, stride)
    else:
        t = expand_dim_n_bottleneck_B(split, bottom, features*2, features, stride)
        t = same_dim_n_bottlneck_B(split, t, features*4, features,  stride)
        r = same_dim_n_bottlneck_B(split, t, features*4, features, stride)
    return r

# The input dim and output dim must be the same
def same_dim_n_bottlneck_B(split, bottom, dim_in, features, stride):
    dim_branch_out = int(math.floor(features * (opt.baseWidth / 64)))
    dim_out = dim_in
    branch_container=list()
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, bottom, dim_branch_out, 1, 1, 0);
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, stride, 1)
        branch_container.append(relu2)
    comb = L.Concat(*branch_container, in_place=True)
    scale_comb, relu_comb = conv_BN_scale_relu(split, comb, dim_out, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, bottom, dim_out, 1, stride, 0)
    return L.Eltwise(relu_comb, relu_i, operation = P.Eltwise.SUM)

def expand_dim_n_bottleneck_B(split, bottom, dim_in, features, stride):
    dim_branch_out = int(math.floor(features * (opt.baseWidth / 64)))
    dim_out = dim_in * 4
    branch_container=list()
    for i in range(opt.cardinalty):
        scale1, relu1 = conv_BN_scale_relu(split, bottom, dim_branch_out, 1, 1, 0);
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, stride, 1)
        branch_container.append(relu2)
    comb = L.Concat(*branch_container, in_place=True)
    scale_comb, relu_comb = conv_BN_scale_relu(split, comb, dim_out, 1, 1, 0)
    scale_i, relu_i = conv_BN_scale_relu(split, bottom, dim_out, 1, stride, 0)
    return L.Eltwise(relu_comb, relu_i, operation = P.Eltwise.SUM)


if __name__ == "__main__":
    global opt
    opt = opts()
    n = (opt.depth - 2) / 9
    print (" | ResNet-" + str(opt.depth) + " " + opt.dataset)

    # Write models to protobuf
    with open(TRAIN_NET_PATH, "w") as f:
        f.write(str(ResNext('train', n)))
    with open(TEST_NET_PATH, "w") as f:
        f.write(str(ResNext('test', n)))


