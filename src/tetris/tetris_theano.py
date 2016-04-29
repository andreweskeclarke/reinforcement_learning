from __future__ import print_function
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T

class DenseLayer(object):
    def __init__(self, n_input, n_output, activation=T.tanh):
        W = np.random.randn(n_output, n_input)
        b = np.ones(n_output)
        self.W = theano.shared(value=W.astype(theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=b.reshape(n_output, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        return self.activation(T.dot(self.W, x) + self.b)

class Model(object):

    def __init__(self, layers=None, learning_rate=0.1):
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
        input = T.matrix('input')
        x = x[:, np.newaxis]
        predict_function = theano.function([input], self.output(input))
        y = predict_function(x)
        print('Y shaoe {}'.format(y.shape))
        return y

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)

    def gradient_updates(self, cost, params):
        return [(param, param - self.learning_rate*T.grad(cost, param)) for param in params]

    def compile(self):
        input = T.vector('input')
        target = T.vector('target')
        cost_function = self.squared_error(input, target)
        self.train_function = theano.function([input, target], cost_function,
                                updates=self.gradient_updates(cost_function, self.params))

    def train(self, X, Y, n_epochs):
        learning_rate = 0.01
        epoch = 0
        while epoch < n_epochs:
            for x, y in zip(X, Y):
                current_cost = self.train_function(x, y)
            epoch += 1
        return current_cost

