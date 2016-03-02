from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD, RMSprop
from tetris_game import *
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

BUFFER_SIZE = 500000

class Agent():
    def __init__(self):
        self.init_model()
        self.save_requests = 0
        # Treat as a ring buffer
        self.max_pos = 0
        self.current_pos = 0
        self.n_plays = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,22,10), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE], dtype=np.int16)
        self.states_t1 = np.zeros((BUFFER_SIZE,1,22,10), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE], dtype=np.int32)
        self.interesting_indexes = list()
        self.current_episode_length = 0
        self.training_runs = 0

    def exploit(self):
        # Simple linear exploit, max at 90% exploitations
        if self.n_plays > 100000:
            return random.random() < 0.50 + (0.30 * ((self.n_plays - 100000)/(float(300000))))
        if self.n_plays > 400000:
            return random.random() < 0.80
        return random.random() < 0.50

    def choose_action(self, state):
        state = (state > 0).astype(np.int8)
        if self.exploit():
            vals = self.model.predict(np.array(state, ndmin=4), verbose=0)
            if random.random() < 0.005:
                print('Some predicted values for a board. Trained against {} examples so far.'.format(self.training_runs))
                print(np.array(state, ndmin=4))
                print('[ROTATE_LEFT, ROTATE_RIGHT, MOVE_RIGHT, MOVE_LEFT, MOVE_DOWN, DO_NOTHING]')
                print(vals)
                print(np.argmax(vals))
            choice = np.argmax(vals)
        else:
            choice = random.choice(MOVES_POOL)
        return choice
        
    def handle(self, state0, action, reward, state1):
        # Have inputs vary equally around 0, allows nodes to move in +/- direction.
        state0 = ((state0 > 0).astype(np.int8) * 2) - 1
        state1 = ((state1 > 0).astype(np.int8) * 2) - 1
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = reward
        self.states_t1[self.current_pos] = state1
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        if self.current_pos == 0:
            self.flush()
        self.n_plays += 1
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        if reward != 0:
            # Backups
            if self.current_episode_length == 0:
                indexes = [(self.current_pos - 1) % BUFFER_SIZE]
            else:
                indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 1 - self.current_episode_length, -1)]
            DISCOUNT = 0.8
            self.interesting_indexes.append(indexes)
            for i, index in enumerate(indexes):
                self.rewards[index] += reward * (DISCOUNT ** i) # TODO: some rounding issues here...
            self.train_on_indexes(indexes)
            self.current_episode_length = 0
        else:
            self.current_episode_length += 1
        self.experience_replay()
        self.save()

    def flush(self):
        self.interesting_indexes = list()

    def save(self):
        self.save_requests += 1
        if self.save_requests == 100000:
            self.save_requests = 0
            name = time.strftime("%m-%dT%H%M%S%Z")
            model_file = 'output/model_{}.json'.format(name)
            weights_file = 'output/weights_{}.h5'.format(name)
            open(model_file, 'w').write(self.model.to_json())
            self.model.save_weights(weights_file, overwrite=True)
            print('Saved: {} and {}'.format(model_file, weights_file))

    def experience_replay(self):
        if len(self.interesting_indexes) > 0:
            indexes = list()
            for i in range(0, N_REPLAYS_PER_ROUND):
                indexes += random.choice(self.interesting_indexes)
            random.shuffle(indexes)
            self.train_on_indexes(indexes)

    def train_on_indexes(self, indexes):
        self.training_runs += len(indexes)
        y = self.model.predict(self.states_t0[indexes], verbose=0)
        DISCOUNT = 0.8
        future_rewards = DISCOUNT*(np.amax(self.model.predict(self.states_t1[indexes], verbose=0), axis=1))
        for i, a in enumerate(self.actions[indexes]):
            y[i][a] = self.rewards[indexes][i] + future_rewards[i]
        self.model.train_on_batch(self.states_t0[indexes], y)

    def init_model(self):
    #    self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
    #    self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))
         self.model = Sequential()
         self.model.add(Convolution2D(16, 3, 3, 
                             activation='tanh', 
                             init='he_uniform',
                             input_shape=(1,22,10)))
         self.model.add(Convolution2D(32, 4, 4, 
                             activation='tanh', 
                             init='he_uniform'))
         self.model.add(Flatten())
         self.model.add(Dense(256, activation='relu', init='he_uniform'))
         self.model.add(Dense(len(POSSIBLE_MOVES), activation='linear', init='he_uniform'))
         optim = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
         self.model.compile(loss='mse', optimizer=optim)

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
