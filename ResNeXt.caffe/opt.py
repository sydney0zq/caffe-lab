#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-16

Description: Argument parameter parser. 
"""

from __future__ import print_function

import argparse

def opts():
    parser = argparse.ArgumentParser(description = "Caffe ResNext Training Script\n")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Options: cifar10 | cifar100", choices=["cifar10", "cifar100"])
    #parser.add_argument("--nEpochs", type=int, default=300, help="Number of total epochs to run")
    parser.add_argument("--batch_size", type=int, default=32, help="mini-batch size, default 32")
    parser.add_argument("--netType", type=str, default="resnext", help="Options: resnext", choices=["resnext"]) 
    parser.add_argument("--bottleneckType", type=str, default="resnext_B", help="Options: resnet | resnext_B | resnext_C", 
                                                                    choices=["resnet", "resnext_B", "resnext_C"]);
    parser.add_argument("--depth", type=int, default=29, help="ResNext depth: 29 | 38 | 47 | 56 | 101", choices=[29, 38, 47, 56, 101])
    parser.add_argument("--baseWidth", type=int, default=4, help="ResNext base width", choices=[64, 40, 24, 14, 4])
    parser.add_argument("--cardinalty", type=int, default=32, help="ResNext cardinality", choices=[1, 2, 4, 8, 32])
    args = parser.parse_args()
    return args







