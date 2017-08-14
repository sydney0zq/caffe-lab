#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-14 

Description: 
"""

import caffe
from caffe import layers as L, params as P

def LeNet(lmdb, batch_size):
    # A series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type="xavier"))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type="xavier"))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1   = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type="xavier"))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type="xavier"))
    n.loss  = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open("./lenet_auto_train.protobuf", "w") as f:
    f.write(str(LeNet("./mnist/mnist_train_lmdb", 64)))

with open("./lenet_auto_test.protobuf", "w") as f:
    f.write(str(LeNet("./mnist/mnist_test_lmdb", 100)))





