#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-16

Description: 
    Decompose input dimension to C cardinalities, each of them
    are `dim_branch_out` dimension.
    
    Pay attention to stride.
"""

from __future__ import print_function

import caffe
from caffe import layers as L, params as P, to_proto
from conv import conv_BN_scale_relu


def decompose(split, bottom, dim_in, dim_branch_out, cardinality):
    branch_container = list()
    for i in range(cardinality):
        # torch#75: s:add(Convolution(nInputPlane,d,1,1,1,1,0,0))
        scale1, relu1 = conv_BN_scale_relu(split, bottom, dim_branch_out, 1, 1, 0);
        # torch#76: s:add(Convolution(d,d,3,3,stride,stride,1,1))
        # NOTE: I don't think stride is needed, as we need to keep dim the same
        scale2, relu2 = conv_BN_scale_relu(split, relu1,  dim_branch_out, 3, 1, 1)
        branch_container.append(relu2)
    comb = L.Concat(*branch_container, in_place=True)
    return comb
    
