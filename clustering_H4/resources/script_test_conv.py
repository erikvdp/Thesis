"""

script to test theano convolutions on raw waveform data

"""

import numpy as np
import sys
import os
import h5py
import shutil

import theano.tensor as T
import theano
from theano.tensor.signal.conv import conv2d

part_number = 100
SOURCE_PATH_MASK = "/mnt/storage/data/msd/dataset_raw/dataset_raw_%04d.h5"
source_path = SOURCE_PATH_MASK % part_number
source_file = h5py.File(source_path, 'r')

X = source_file['X']
num_clips, clip_len = X.shape
X = X[:10]

def test_conv(X, music_shape, filter_shape, subsample):
    """
    X: Shared variable
    """

    W = T.tensor3('W')
    output = conv2d(X, W, image_shape=music_shape, filter_shape=filter_shape, subsample=subsample)
    f = theano.function([W], output)

    ' now test: '
    try:
        W = np.random.rand(*filter_shape).astype(np.float32)
        f(W)
        return 1
    except Exception, err:
        print 'ERROR: %s\n' % str(err)
        return 0

X_shared = theano.shared(X.astype(np.float32))
music_shape = X.shape
for n_filters in [128]:
    for filter_length in [128]:
        for subsample in [32]:
            print n_filters, filter_length, subsample
            res = test_conv(X_shared, music_shape, (n_filters, 1, filter_length), (1, subsample))
            print 'score!' if res else 'fail!'
            print '\n'
