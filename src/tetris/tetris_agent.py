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

N_REPLAYS_PER_ROUND = 5

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

BUFFER_SIZE = 1000000

class Agent():
    def __init__(self):
        self.init_model()
        self.save_requests = 0
        # Treat as a ring buffer
        self.max_pos = 0
        self.current_pos = 0
        self.n_plays = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,22,10), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE])
        self.states_t1 = np.zeros((BUFFER_SIZE,1,22,10), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE])

    def exploit(self):
        # Simple linear exploit, max at 90% exploitations
        return random.random() < 0.80

    def choose_action(self, state):
        if self.exploit():
            vals = self.model.predict(np.array(state, ndmin=4), verbose=0)
            if random.random() < 0.0001:
                print('Some predicted values for a board:')
                print(np.array(state, ndmin=4))
                print('[ROTATE_LEFT, ROTATE_RIGHT, MOVE_RIGHT, MOVE_LEFT, MOVE_DOWN, DO_NOTHING]')
                print(vals)
            return np.argmax(vals)
        return random.choice(MOVES_POOL)
        
    def handle(self, state0, action, reward, state1):
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = reward
        self.states_t1[self.current_pos] = state1
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.n_plays += 1
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        if reward > 0 and self.n_plays > 4:
            # Prioritized Sweeping
            indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 3, -1)]
            for i, index in enumerate(indexes):
                self.rewards[index] += reward * (1/(2+index))
            self.train_on_indexes(indexes)
        self.experience_replay()
        self.save()

    def save(self):
        self.save_requests += 1
        if self.save_requests == 50000:
            self.save_requests = 0
            name = time.strftime("%m-%dT%H%M%S%Z")
            model_file = 'output/model_{}.json'.format(name)
            weights_file = 'output/weights_{}.h5'.format(name)
            open(model_file, 'w').write(self.model.to_json())
            self.model.save_weights(weights_file, overwrite=True)
            print('Saved: {} and {}'.format(model_file, weights_file))

    def experience_replay(self):
        indexes = np.random.randint(0, self.max_pos, N_REPLAYS_PER_ROUND)
        self.train_on_indexes(indexes)

    def train_on_indexes(self, indexes):
        y = self.model.predict(self.states_t0[indexes], verbose=0)
        future_rewards = np.amax(self.model.predict(self.states_t1[indexes], verbose=0), axis=1)
        for i, a in enumerate(self.actions[indexes]):
            y[i][a] = self.rewards[indexes][i] + future_rewards[i]
        self.model.train_on_batch(self.states_t0[indexes], y)

    def init_model(self):
        self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
        self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))
    #     self.model = Sequential()
    #     # 32 Convolution filters of size 2x2 each
    #     self.model.add(Convolution2D(32, 2, 2, 
    #                                  input_shape=(1,22,10), 
    #                                  activation='relu'))
    #     # Second convolution layer
    #     self.model.add(Convolution2D(64, 4, 4, 
    #                                  input_shape=(32,21,9), 
    #                                  activation='relu'))
    #     # Flatten to a single vector of inputs
    #     self.model.add(Flatten())
    #     # Dense hidden layer
    #     self.model.add(Dense(64, activation='relu', init='uniform'))
    #     self.model.add(Dense(64, activation='relu', init='uniform'))
    #     self.model.add(Dense(len(POSSIBLE_MOVES), activation='linear', init='uniform'))
    #     self.model.compile(loss='mse', optimizer='rmsprop')

class GreedyAgent(Agent):
    def __init__(self, model_path=None, weights_path=None):
        threading.Thread.__init__(self)
        self.model_path = model_path
        self.weights_path = weights_path
        self.init_model()
        self.replay_memory = []

    def init_model(self):
        if self.model_path:
            self.model = model_from_json(open(self.model_path).read())
        else:
            self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
        if self.weights_path:
            self.model.load_weights(self.weights_path)
        else:
            self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))

    def choose_action(self, state):
        time.sleep(0.05)
        return random.choice(MOVES_POOL)
        # return np.argmax(self.model.predict(np.array(state, ndmin=4), batch_size=1, verbose=0))

    def handle(self, s0, a, r, s1):
        pass
