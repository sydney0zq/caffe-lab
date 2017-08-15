#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e

EXAMPLE="."
DATA="."
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

~/caffe/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

~/caffe/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
