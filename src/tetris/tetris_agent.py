from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD
from tetris_game import POSSIBLE_MOVES, TIME_PER_TICK
import numpy as np
import queue
import threading
import time
import random

N_REPLAYS_PER_ROUND = 5

class Agent(threading.Thread):
    def __init__(self, actions_q, states_q, print_q):
        threading.Thread.__init__(self)
        self.actions_q = actions_q
        self.states_q = states_q
        self.print_q = print_q
        self.daemon = True
        self.init_model()
        self.replay_memory = []

    def choose_action(self, state):
        if bool(random.getrandbits(1)):
            self.print_q.put('Greedy: True')
            return np.argmax(self.model.predict(np.array([[state]]), batch_size=1, verbose=0))
        self.print_q.put('Greedy: False')
        return random.choice(POSSIBLE_MOVES)
        
    def handle_new_state(self, state):
        self.replay_memory.append(state)

    def run(self):
        while True:
            start = time.time()
            state = self.states_q.get(True)
            self.handle_new_state(state)
            self.experience_replay()
            self.actions_q.put(self.choose_action(state[3]), True)
            self.print_q.put('{}'.format(time.time() - start))

    def experience_replay(self):
        self.train(random.choice(self.replay_memory))

    def train(self, sample):
        # SARS - [State_1, Action, Reward, State_2]
        action = sample[1]
        immediate_reward = sample[2]
        future_reward = np.max(self.model.predict(np.array([[sample[3]]]), batch_size=1, verbose=0))
        reward = immediate_reward + future_reward
        x = sample[0]
        y = self.model.predict(np.array([[sample[0]]]), verbose=0)
        y[0][action] = reward
        self.model.train_on_batch(np.array([[x]]), y)

    def init_model(self):
        self.model = Sequential()
        # 32 Convolution filters of size 4x4 each
        self.model.add(Convolution2D(32, 4, 4, 
                                     input_shape=(1,24,10), 
                                     activation='relu'))
        # Pool the vertical dimension, halving the size of inputs without lossing horizontal granularity
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        # Flatten to a single vector of inputs
        self.model.add(Flatten())
        # Dense hidden layer
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        # Output actions, whose value we are estimating
        self.model.add(Dense(len(POSSIBLE_MOVES)))
        self.model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
