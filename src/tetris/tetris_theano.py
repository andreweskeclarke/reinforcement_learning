from __future__ import print_function
import time
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
import math

class DenseLayer(object):
    def __init__(self, n_input, n_output):
        m = 1/ math.sqrt(n_input)
        W = (m * np.random.randn(n_output, n_input)).astype(theano.config.floatX)
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.W, self.b]

    def output(self, x):
        return T.tanh(T.dot(self.W, x) + self.b)

class Flatten(object):
    def __init__(self):
        self.params = []

    def output(self, x):
        return T.flatten(x)

class Conv2DLayer(object):
    def __init__(self, n_filters, n_cols, n_rows):
        self.output_shape = (n_filters, 1, n_cols, n_rows)
        m = 1 / math.sqrt(n_filters * n_cols * n_rows)
        W = (m * np.random.randn(*self.output_shape)).astype(theano.config.floatX)
        b = np.zeros((n_filters,)).astype(theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.W, self.b]

    def output(self, x):
        x = T.reshape(x, (1,1,20,10))
        return T.tanh(conv2d(x, self.W) + self.b.dimshuffle('x', 0, 'x', 'x'))

class Model(object):

    def __init__(self, layers=None, learning_rate=0.01):
        self.layers = []
        self.params = []
        self.learning_rate = learning_rate
        if layers is not None:
            self.layers = layers
            for l in layers:
                self.params += l.params

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        
    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def predict(self, x):
        return self.predict_function(x)

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)

    def sgd(self, cost, params, lr=0.1):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    def compile(self):
        input = T.vector('input')
        target = T.vector('target')
        cost_function = self.squared_error(input, target)
        self.train_function = theano.function([input, target],
                                cost_function,
                                updates=self.sgd(cost_function, self.params, self.learning_rate))
        output_function = self.output(input)
        self.predict_function = theano.function([input], output_function)

    def train(self, X, Y, n_epochs):
        start = time.time()
        epoch = 0
        print(X.shape)
        print(Y.shape)
        while epoch < n_epochs:
            for x, y in zip(X, Y):
                current_cost = self.train_function(x, y)
            epoch += 1
        print('Training {} samples over {} epochs took {}s'.format(X.shape[0], n_epochs, time.time() - start))
        print(self.params)
        return current_cost
