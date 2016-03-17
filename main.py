#! coding=utf-8

import gzip, cPickle, timeit
import theano
import theano.tensor as T
import numpy as np
import rbm
from PIL import Image
import utils, logistic


def shared_data(data):
    X, y = data
    X_shared = theano.shared(np.array(X, dtype=theano.config.floatX), borrow=True)
    y_shared = theano.shared(np.array(y, dtype=theano.config.floatX), borrow=True)
    return X_shared, T.cast(y_shared, 'int32')


def main():
    print 'load data...'
    with gzip.open('./dataset/mnist.pkl.gz') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    train_set_x, train_set_y = shared_data(train_set)
    # model params
    learning_rate = 0.01
    batch_size = 20
    epoches = 100
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    x = T.matrix('input')
    index = T.lscalar('index')
    rng = np.random.RandomState(1234)
    print 'initial params...'
    RBM = rbm.RBM(x, n_visiable=784, n_hidden=500, numpy_rng=rng)
    cost, updates = RBM.cost_updates(lr=learning_rate, k_step=1)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size]
        }
    )
    print 'training...'
    for epoch in xrange(epoches):
        t1 = timeit.default_timer()
        train_cost = [train_model(i) for i in xrange(n_train_batches)]
        print 'epoch = {}, cross-entroy = {}, time = {}'.format(epoch, np.mean(train_cost), timeit.default_timer() - t1)
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        if epoch % 5 == 0:
            image = Image.fromarray(
                utils.tile_raster_images(
                    X=RBM.W.get_value(borrow=True).T,
                    img_shape=(28, 28),
                    tile_shape=(10, 10),
                    tile_spacing=(1, 1)
                )
            )
            image.save('./result_image/filters_at_epoch_%i.png' % epoch)
    with open('model.pkl', 'w') as f:
        cPickle.dump(RBM, f)
        print 'save model'


if __name__ == '__main__':
    main()
