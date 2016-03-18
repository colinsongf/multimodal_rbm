import load_data, logistic, theano
import theano.tensor as T
import numpy as np

x = T.matrix('x')
y = T.ivector('y')
index = T.lscalar('index')
rng = np.random.RandomState(1234)

hidden_layer = logistic.Hidden_layer(x, 784, 500)
soft_max = logistic.Softmax_layer(hidden_layer.outputs, 500, 10)
cost = soft_max.nagetive_likehood(y)
params = hidden_layer.params + soft_max.params
gparams = T.grad(cost, params)
updates = [(param, param - 0.01*gparam) for param, gparam in zip(params, gparams)]
datasets = load_data.load_data('./dataset/mnist.pkl.gz')
[train_set_x, train_set_y] = datasets[0]
n_train_batches = train_set_x.get_value(borrow=True).shape[0]/20

train_model = theano.function([index], cost, updates=updates, givens={
    x:train_set_x[20*index:(index+1)*20],
    y:train_set_y[20*index:(index+1)*20]
})

for epoch in xrange(20):
    train_model_cost = [train_model(i) for i in xrange(n_train_batches)]
    print np.mean(train_model_cost)