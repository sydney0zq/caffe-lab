#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-16 

Description: 
    Implementation of convolution, scale and relu.

    Notice that caffe needs scale layer to learn parameter, while
    BN layer cannot learn gamma/beta.
"""

from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto

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
