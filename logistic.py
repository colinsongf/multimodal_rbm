#! coding=utf-8

import theano
import theano.tensor as T
import numpy as np


class Softmax_layer:
    def __init__(self, inputs, n_in, n_out, rng):
        self.inputs = inputs
        w_val = np.array(rng.uniform(np.sqrt(4. / (n_in + n_out)), np.sqrt(4. / n_out + n_in), size=(n_in, n_out)),
                         theano.config.floatX)
        self.W = theano.shared(value=w_val, name='w', borrow=True)
        b_val = np.zeros(n_out, theano.config.floatX)
        self.b = theano.shared(value=b_val, name='b', borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.nnet.softmax(T.dot(self.inputs, self.W) + self.b)
        self.pred_y = T.argmax(self.outputs, axis=1)

    def nagetive_likehood(self, y):
        return -T.mean(T.log(self.outputs)[T.arange(y.shape[0]), y])

    def error(self, y):
        return T.mean(T.neq(y, self.pred_y))


class Hidden_layer:
    def __init__(self, inputs, n_in, n_out, rng):
        self.inputs = inputs
        w_val = np.array(rng.uniform(np.sqrt(4. / (n_in + n_out)), np.sqrt(4. / n_out + n_in), size=(n_in, n_out)),
                         theano.config.floatX)
        self.W = theano.shared(value=w_val, name='w', borrow=True)
        b_val = np.zeros(n_out, theano.config.floatX)
        self.b = theano.shared(value=b_val, name='b', borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.nnet.sigmoid(T.dot(self.inputs, self.W) + self.b)
