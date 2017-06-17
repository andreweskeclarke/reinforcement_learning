from __future__ import print_function
import time
import os
import sys
import timeit
import pickle
import theano
import numpy as np
import theano.tensor as T
from tetris_game import POSSIBLE_MOVES
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import downsample
import math


class DenseLayer(object):
    def __init__(self, n_input, n_output):
        m = 1 / math.sqrt(n_input)
        W = (m * np.random.randn(n_output, n_input)).astype(theano.config.floatX)
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.W, self.b]

    def output(self, x, a):
        return T.tanh(T.dot(self.W, x) + self.b)


class Flatten(object):
    def __init__(self):
        self.params = []

    def output(self, x, a):
        return x.flatten()


class Conv2DLayer(object):
    def __init__(self, n_filters, n_cols, n_rows, n_inputs, width, height):
        self.output_shape = (n_filters, n_inputs, n_cols, n_rows)
        m = 1 / math.sqrt(n_filters * n_cols * n_rows)
        W = (m * np.random.randn(*self.output_shape)).astype(theano.config.floatX)
        b = np.zeros((n_filters,)).astype(theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.W, self.b]
        self.width = width
        self.height = height
        self.n_inputs = n_inputs

    def output(self, x, a):
        x = T.reshape(x, (-1, self.n_inputs, self.height, self.width))
        return T.tanh(conv2d(x, self.W) + self.b.dimshuffle('x', 0, 'x', 'x'))


class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p
        self.params = []

    def output(self, x, a):
        p = self.p
        srng = RandomStreams()
        if p > 0:
            retain_prob = 1 - p
            x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
            x /= retain_prob
        return x


class StateAndActionMerge(object):
    def __init__(self):
        self.params = []

    def output(self, x, a):
        return T.concatenate([x.flatten(), a.flatten()])


class ActionAdvantageMerge(object):
    def __init__(self):
        self.params = []

    def output(self, A, V):
        return V.repeat(len(POSSIBLE_MOVES)) + (A - T.mean(A).repeat(len(POSSIBLE_MOVES)))


class Split(object):
    def __init__(self, branch1, branch2, merger):
        self.branch1 = Model(branch1)
        self.branch2 = Model(branch2)
        self.merger = merger
        self.params = self.branch1.params + self.branch2.params

    def output(self, x, a):
        x1 = self.branch1.output(x, a)
        x2 = self.branch2.output(x, a)
        return self.merger.output(x1, x2)


class MaxPooling(object):
    def __init__(self, shape):
        self.params = []

    def output(self, x, a):
        return downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)


class Model(object):
    def __init__(self, layers=None):
        self.layers = []
        self.params = []
        if layers is not None:
            self.layers = layers
            for l in layers:
                self.params += l.params

    def copy(self):
        model = Model(self.layers)
        model.train_function = self.__copy_theano(self.train_function)
        model.predict_function = self.__copy_theano(self.predict_function)
        return model

    def __copy_theano(self, theano_f):
        sys.setrecursionlimit(5000)
        pickle.dump(theano_f,
            open('/tmp/theano_model.p', 'wb'), 
            protocol=pickle.HIGHEST_PROTOCOL)
        copied_f = pickle.load(open('/tmp/theano_model.p', 'rb'))
        sys.setrecursionlimit(1000)
        return copied_f

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        
    def output(self, x, a):
        for layer in self.layers:
            x = layer.output(x, a)
        return x

    def predict(self, x, a):
        return self.predict_function(x, a)

    def squared_error(self, x, a, y):
        result, updates = theano.scan(
            fn=lambda xx, aa, yy: (self.output(xx, aa) - yy) ** 2,
            outputs_info=None,
            sequences=[x,a,y]
            )
        return T.sum(result)
        # return T.sum((self.output(x, a) - y) ** 2)

    def sgd(self, cost, params, lr=0.0001):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    def compile(self):
        x_train = T.tensor4('x_train')
        actions_train = T.matrix('actions_train')
        y_train = T.matrix('y_train')
        cost_function = self.squared_error(x_train, actions_train, y_train)
        self.train_function = theano.function([x_train, actions_train, y_train],
                                cost_function,
                                updates=self.sgd(cost_function, self.params),
                                on_unused_input='ignore',
                                allow_input_downcast=True)
        x_pred = T.tensor3('x_pred')
        actions_pred = T.vector('actions_pred')
        output_function = self.output(x_pred, actions_pred)
        self.predict_function = theano.function([x_pred, actions_pred],
                                                output_function,
                                                on_unused_input='ignore',
                                                allow_input_downcast=True)
        return self

    def train(self, X, A, Y, n_epochs):
        self.train_function(X, A, Y)
