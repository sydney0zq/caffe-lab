#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Mail: theodoruszq@gmail.com, HUST
Version: 0.0
Date: 2017-08-22

Description:
"""

import matplotlib.pyplot as plt
import numpy as np
import time

date = time.strftime("%Y-%m-%d", time.localtime())
LOG_TEST_FILE="tmp_test.log"


train_data = []
for line in open(LOG_TEST_FILE, "r"):
    d = line.split(" ")[-1].rstrip()
    train_data.append(d)

print(max(train_data))




