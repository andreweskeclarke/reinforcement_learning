from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD
from tetris_game import POSSIBLE_MOVES, TIME_PER_TICK, MOVES_POOL
from collections import deque
import os
import glob
import numpy as np
import queue
import threading
import time
import random
import math

N_REPLAYS_PER_ROUND = 20

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

BUFFER_SIZE = 10000000

class Agent():
    def __init__(self):
        self.init_model()
        self.save_requests = 0
        # Treat as a ring buffer
        self.max_pos = 0
        self.current_pos = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,22,10))
        self.actions = np.zeros([BUFFER_SIZE])
        self.states_t1 = np.zeros((BUFFER_SIZE,1,22,10))
        self.rewards = np.zeros([BUFFER_SIZE])

    def choose_action(self, state):
        if bool(random.getrandbits(1)):
            return np.argmax(self.model.predict(np.array(state, ndmin=4), batch_size=1, verbose=0))
        return random.choice(MOVES_POOL)
        
    def handle(self, state0, action, reward, state1):
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = reward
        self.states_t1[self.current_pos] = state1
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        self.experience_replay()
        if reward == -10000:
           self.save()

    def save(self):
        self.save_requests += 1
        if self.save_requests == 500:
            self.save_requests = 0
            name = time.strftime("%m-%dT%H%M%S%Z")
            model_file = 'output/model_{}.json'.format(name)
            weights_file = 'output/weights_{}.h5'.format(name)
            open(model_file, 'w').write(self.model.to_json())
            self.model.save_weights(weights_file, overwrite=True)
            print('Saved: {} and {}'.format(model_file, weights_file))

    def experience_replay(self):
        start = time.time()
        indexes = np.random.randint(0, self.max_pos, N_REPLAYS_PER_ROUND)
        y = self.model.predict(self.states_t0[indexes], verbose=0)
        future_rewards = np.amax(self.model.predict(self.states_t1[indexes], verbose=0), axis=1)
        for i, a in enumerate(self.actions[indexes]):
            y[i][a] = self.rewards[indexes][i] + future_rewards[i]
        self.model.train_on_batch(self.states_t0[indexes], y)

    def init_model(self):
        self.model = Sequential()
        # 32 Convolution filters of size 2x2 each
        self.model.add(Convolution2D(32, 2, 2, 
                                     input_shape=(1,22,10), 
                                     activation='relu'))
        self.model.add(Dropout(0.25))
        # Second convolution layer
        self.model.add(Convolution2D(64, 4, 4, 
                                     input_shape=(32,21,9), 
                                     activation='relu'))
        self.model.add(Dropout(0.25))
        # Flatten to a single vector of inputs
        self.model.add(Flatten())
        # Dense hidden layer
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        # Output actions, whose value we are estimating
        self.model.add(Dense(len(POSSIBLE_MOVES)))
        self.model.add(Activation('linear'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

class GreedyAgent(Agent):
    def __init__(self):
        threading.Thread.__init__(self)
        self.init_model()
        self.replay_memory = []

    def init_model(self):
        self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
        self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))

    def choose_action(self, state):
        return np.argmax(self.model.predict([state], batch_size=1, verbose=0))

    def handle(self, s0, a, r, s1):
        pass
