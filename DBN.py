#! coding = utf-8
import theano, rbm, logistic, load_data
import theano.tensor as T
import numpy as np
import timeit, cPickle
from theano.sandbox.rng_mrg import MRG_RandomStreams
class DBN:
    def __init__(self, numpt_rng, theano_rng=None, n_in=784, hidden_layers_size=[500,500], n_out=10):
        self.sigmodi_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_size)

        assert self.n_layers >= 0
        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpt_rng.randint(2**30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            if i==0:
                input_size = n_in
            else:
                input_size = hidden_layers_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmodi_layers[-1].output

            sigmoid_layer = logistic.HiddenLayer(
                rng=numpt_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_size[i],
            )
            self.sigmodi_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            rbm_layer = rbm.RBM(
                inputs=layer_input,
                n_visiable=input_size,
                n_hidden=hidden_layers_size[i],
                numpy_rng=numpt_rng,
                theano_rng=theano_rng,
                W=sigmoid_layer.W,
                h_bias=sigmoid_layer.b
            )
            self.rbm_layers.append(rbm_layer)

        self.logLayer = logistic.Softmax_layer(
            inputs=self.sigmodi_layers[-1].output,
            n_in=hidden_layers_size[-1],
            n_out=n_out,
        )
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.nagetive_likehood(self.y)
        self.errors = self.logLayer.error(self.y)

    def pretrainging_function(self, train_set_x, batch_size, k):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        n_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
        batch_begin = index*batch_size
        batch_end = batch_begin + batch_size
        pretrain_fns = []

        for rbm in self.rbm_layers:
            cost, updates = rbm.cost_updates(learning_rate, k_step=k)
            fn = theano.function(
                inputs=[index, learning_rate],
                outputs=cost,
                updates=updates,
                givens={
                    self.x:train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_function(self, datasets, batch_size, learning_rate):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost, self.params)
        updates = [(param, param - learning_rate*g_param) for param, g_param in zip(self.params, gparams)]
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x:train_set_x[index*batch_size:(index+1)*batch_size],
                self.y:train_set_y[index*batch_size:(index+1)*batch_size],
            }
        )
        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x:test_set_x[index*batch_size:(index+1)*batch_size],
                self.y:test_set_y[index*batch_size:(index+1)*batch_size],
            }
        )
        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x:valid_set_x[index*batch_size:(index+1)*batch_size],
                self.y:valid_set_y[index*batch_size:(index+1)*batch_size],
            }
        )
        return train_fn, valid_score_i, test_score_i
def test_DBN(finetune_lr=0.1, pretraining_epoches=100, pretrain_lr=0.01, k=1, \
             training_epoches=1000, dataset='./dataset/mnist.pkl.gz', batch_size=20):
    datasets = load_data.load_data((dataset))
    [train_set_x, train_set_y] = datasets[0]
    [valid_set_x, valid_set_y] = datasets[1]
    [test_set_x, test_set_y] = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size
    numpy_rng = np.random.RandomState(1234)

    print 'building the model...'
    dbn = DBN(numpt_rng=numpy_rng,
              n_in=784,
              hidden_layers_size=[1000, 1000, 1000],
              n_out=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print 'get the pretraining functions...'
    pretraining_fns = dbn.pretrainging_function(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    start_time = timeit.default_timer()
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epoches):
            pretraining_cost = [pretraining_fns[i](j, pretrain_lr) for j in xrange(n_train_batches)]
            print 'Pre-training layser {}, epoch {}, cost {}'.format(i, epoch, np.mean(pretraining_cost))
    end_time = timeit.default_timer() - start_time

    print 'The pretraining code ran for {} mins'.format(end_time/60.)

    ######################
    # FINETUNE THE MODEL #
    ######################

    print 'getting the finetuning function...'
    train_fn, valid_score, test_score = dbn.build_finetune_function(datasets,
                                                                    batch_size=batch_size,
                                                                    learning_rate=0.1)
    best_acc = np.inf

    for epoch in xrange(training_epoches):
        t1 = timeit.default_timer()
        avg_cost_train = np.mean([train_fn(i) for i in xrange(n_train_batches)])
        valid_sc = np.mean([valid_score(i) for i in xrange(n_valid_batches)])
        print 'epoch = {}, train_cost = {}, valid_accury = {}, time = {}'.format(epoch, avg_cost_train, np.mean(valid_sc), timeit.default_timer()-t1)
        if valid_sc < best_acc:
            best_acc = valid_sc
            with open('best_model.pkl', 'w') as f:
                cPickle.dump(dbn, f)
            print 'save model with best auccracy = {}'.format(best_acc)



if __name__ == '__main__':
    test_DBN(finetune_lr=0.1, pretraining_epoches=50, pretrain_lr=0.1, k=1, \
             training_epoches=200, dataset='./dataset/mnist.pkl.gz', batch_size=20)
