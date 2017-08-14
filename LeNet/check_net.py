#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-14 

Description: 
Check model in protobuf.
"""

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver("./lenet_auto_solver.protobuf")
# Each output is (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

# Just print the weight sizes(omit the biases)
print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

print solver.net.forward()    #train net
print solver.test_nets[0].forward()   #test net, there can be more than one





