from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD
from tetris_game import POSSIBLE_MOVES, TIME_PER_TICK
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

class Agent(threading.Thread):
    def __init__(self, actions_q, states_q, print_q):
        threading.Thread.__init__(self)
        self.actions_q = actions_q
        self.states_q = states_q
        self.print_q = print_q
        self.daemon = True
        self.init_model()
        self.replay_memory = list()
        self.save_requests = 0

    def choose_action(self, state):
        if bool(random.getrandbits(1)):
            return np.argmax(self.model.predict(np.array(state, ndmin=4), batch_size=1, verbose=0))
        return random.choice(POSSIBLE_MOVES)
        
    def handle_new_state(self, state):
        self.replay_memory.append(state)

    def run(self):
        while True:
            start = time.time()
            state = self.states_q.get()
            self.handle_new_state(state)
            self.experience_replay()
            if state[REWARD_INDEX] == -10000:
               self.save()
            else:
                self.actions_q.put(self.choose_action(state[STATE1_INDEX]))

    def save(self):
        self.save_requests += 1
        if self.save_requests == 700:
            print('Saving a model...')
            self.save_requests = 0
            name = time.strftime("%m-%dT%H%M%S%Z")
            model_file = 'output/model_{}.json'.format(name)
            weights_file = 'output/weights_{}.h5'.format(name)
            open(model_file, 'w').write(self.model.to_json())
            self.model.save_weights(weights_file, overwrite=True)
            print('Saved: {} and {}'.format(model_file, weights_file))

    def experience_replay(self):
        if len(self.replay_memory) > N_REPLAYS_PER_ROUND:
            self.train(random.sample(self.replay_memory, N_REPLAYS_PER_ROUND))

    def train(self, samples):
        start = time.time()
        future_rewards = np.amax(self.model.predict(np.stack([s[STATE1_INDEX] for s in samples], axis=0), verbose=0), axis=1)
        y = self.model.predict(np.stack([s[STATE0_INDEX] for s in samples], axis=0), verbose=0)
        for i, a in enumerate(map(lambda s:s[ACTION_INDEX], samples)):
            y[i][a] = samples[i][REWARD_INDEX] + future_rewards[i]
        self.model.train_on_batch(np.stack([s[STATE0_INDEX] for s in samples]), y)

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
    def __init__(self, actions_q, states_q, print_q):
        threading.Thread.__init__(self)
        self.actions_q = actions_q
        self.states_q = states_q
        self.print_q = print_q
        self.daemon = True
        self.init_model()
        self.replay_memory = []

    def init_model(self):
        self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
        self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))

    def choose_action(self, state):
        return np.argmax(self.model.predict([state], batch_size=1, verbose=0))
        
    def handle_new_state(self, state):
        pass

    def experience_replay(self):
        pass

    def train(self, sample):
        pass

    def save(self):
        pass
