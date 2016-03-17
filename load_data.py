#! coding=utf-8

import gzip, cPickle, theano
import theano.tensor as T
import numpy as np
def shared_data(data):
    X, y = data
    X_shared = theano.shared(np.array(X, dtype=theano.config.floatX), borrow=True)
    y_shared = theano.shared(np.array(y, dtype=theano.config.floatX), borrow=True)
    return X_shared, T.cast(y_shared, 'int32')
def load_data(path):
    with gzip.open(path, 'r') as f:
        datasets = cPickle.load(f)
    train_set_x, train_set_y = shared_data(datasets[0])
    valid_set_x, valid_set_y = shared_data(datasets[1])
    test_set_x, test_set_y = shared_data(datasets[2])
    return [
        [train_set_x, train_set_y],
        [valid_set_x, valid_set_y],
        [test_set_x, test_set_y]
    ]
