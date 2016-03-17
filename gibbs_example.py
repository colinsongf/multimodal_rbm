import theano
import numpy as np
from theano import tensor as T
W_values = np.array([[1, 1, 1], [1, 1, 1]])
W = theano.shared(W_values) # we assume that ``W_values`` contains the
                            # initial values of your weight matrix
bhid_values = np.array([1, 1, 1])
bvis_values = np.array([1, 1])
bvis = theano.shared(bvis_values)
bhid = theano.shared(bhid_values)

trng = T.shared_randomstreams.RandomStreams(1234)

def OneStep(vsample) :
    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
    hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
    print hmean
    return trng.binomial(size=vsample.shape, n=1, p=vmean,
                         dtype=theano.config.floatX)

sample = theano.tensor.vector()

values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values[-1], updates=updates)

value = gibbs10([1, 0])
