#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-23 

Description: 
    Forked from github: garion9103/impl-pruning-caffemodel.
"""


from __future__ import print_function
import caffe
import sys
sys.dont_write_bytecode = True


if len(sys.argv) < 4:
    print ("Usage: vgg_comp.py <input_deploy_file> <input_model_file> <output_model_file>")
    print ("*" * 50)
    print ("Now load default configuration...\n")
    protobuf = "./model/lenet_train.protobuf"
    orig     = "./model/lenet.caffemodel"
    comp     = "./model/pruned_lenet.caffemodel"
else:
    protobuf = sys.argv[1]
    orig     = sys.argv[2]  # Origin model
    comp     = sys.argv[3]  # Pruned model

caffe.set_mode_gpu()
net = caffe.Net(protobuf, orig, caffe.TEST)
#layers = filter(lambda x: "conv" in x or "fc" in x or "ip" in x, net.params.keys())
#print (layers)
#print (net.params.keys())






