#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-19

Description:
    Plot curves from pure training or testing log files.
"""


import os

import matplotlib.pyplot as plt
import numpy as np

LOG_TRAIN_FILE="tmp_train.log"
PNG_TRAIN_FILE="tmp_train.png"
LOG_TEST_FILE="tmp_test.log"
PNG_TEST_FILE="tmp_test.png"

def plot_and_save(list_data, fn):
    plt.figure(figsize=(24,12))
    plt.plot(range(len(list_data)), list_data, 'ro')
    plt.plot(range(len(list_data)), np.ones(len(list_data)))
    plt.plot(range(len(list_data)), 0.9 * np.ones(len(list_data)))
    plt.plot(range(len(list_data)), 0.8 * np.ones(len(list_data)))
    plt.plot(range(len(list_data)), 0.7 * np.ones(len(list_data)))
    plt.savefig(fn, dpi=300)
    plt.clf()

train_data = []
i = 0
for line in open(LOG_TRAIN_FILE, "r"):
    d = line.split(" ")[-1].rstrip()
    i += 1
    #if i % 15 == 0:
    train_data.append(d)

plot_and_save(train_data, PNG_TRAIN_FILE)

test_data = []
for line in open(LOG_TEST_FILE, "r"):
    d = line.split(" ")[-1].rstrip()
    test_data.append(d)

plot_and_save(test_data, PNG_TEST_FILE)




