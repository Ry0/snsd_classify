#!/usr/bin/env sh

CAFFE_ROOT=../../
cd $CAFFE_ROOT
build/tools/compute_image_mean.bin -backend=leveldb ./examples/snsd_classify/snsd_cifar10_train_leveldb ./examples/snsd_classify/snsd_mean.binaryproto
build/tools/caffe train --solver examples/snsd_classify/snsd_cifar10_quick_solver.prototxt