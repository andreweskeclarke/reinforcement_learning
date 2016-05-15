import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tetris_theano
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD, RMSprop, Adam
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
import socket
import sys
import os
import statistics

N_REPLAYS_PER_ROUND = 2000

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

BUFFER_SIZE = 15000
DISCOUNT = 0.4
BROADCAST_PORT = 50006

DEBUG=False

def tbug(msg):
    if DEBUG:
        print(msg)

class StatePrinter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', 0))

    def send_to_websocket(self, state):
        data = bytes("|".join([",".join(map(str, x)) for x in state[0]]) + "\n", 'ascii')
        self.sock.sendto(data, ("<broadcast>", BROADCAST_PORT))

class Agent():
    def __init__(self):
        self.init_model()
        # Treat as a ring buffer
        self.current_pos = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE], dtype=np.int8)
        self.states_t1 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.current_episode_length = 0
        self.current_game_length = 0
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.n_games = 0
        self.state_printer = StatePrinter()
        self.exploiting_turn = bool(random.getrandbits(1))
        self.rolled_over_buffer = False

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return min([0.9, self.n_games / 100])

    def choose_action(self, state):
        if self.exploit():
            action = list([0,0,0])
            q_values = self.model.predict(state, action)
            self.recent_q_values.append(np.max(q_values))
            return np.argmax(q_values)
        return random.choice(POSSIBLE_MOVES)
        
    def last_n_indexes(self, n):
        if n == 0:
            indexes = [(self.current_pos - 1) % BUFFER_SIZE]
        else:
            indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 1 - n, -1)]
        return indexes

    def tick_forward(self):
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.current_game_length += 1
        self.current_episode_length += 1
        if self.current_pos == 0: # Rolled over on indexes
            self.rolled_over_buffer = True

    def game_over(self, total_reward):
        self.state_printer.send_to_websocket(self.states_t1[self.current_pos - 1])
        indexes = self.last_n_indexes(self.current_game_length)
        total_reward_factor = 1.0 + (float(total_reward) * 0.05)
        total_reward_factor = max(0.5, total_reward_factor)
        for i, index in enumerate(indexes):
            self.rewards[index] = self.rewards[index] * total_reward_factor
        self.experience_replay()
        self.current_game_length = 0

    def backup_episode(self, reward):
        indexes = self.last_n_indexes(self.current_episode_length)
        n_ineffective_actions = 0
        for i, index in enumerate(indexes):
            if np.array_equal(self.states_t0[index], self.states_t1[index]):
                n_ineffective_actions += 1
                self.rewards[index] = 0
            else:
                self.rewards[index] += float(reward) * (DISCOUNT**(i - n_ineffective_actions))
        self.current_episode_length = 0

    def on_episode_end(self, reward):
        self.backup_episode(reward)
        self.exploiting_turn = self.exploit()

    def handle(self, state0, action, reward, state1):
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = 0
        self.states_t1[self.current_pos] = state1
        self.tick_forward()
        self.state_printer.send_to_websocket(self.states_t1[self.current_pos - 1])

    def experience_replay(self):
        sys.stdout.flush()
        if self.rolled_over_buffer:
            self.rolled_over_buffer = False
            mask = np.random.rand(BUFFER_SIZE) < 0.3
            X1_train, X2_train, Y_train = self.training_data_for_indexes(mask)
            self.model.train(X1_train, X2_train, Y_train, 1)

    def input_size(self):
        return BOARD_HEIGHT*BOARD_WIDTH + len(POSSIBLE_MOVES)

    def output_size(self):
        return BOARD_HEIGHT*BOARD_WIDTH

    def training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        rewards = self.rewards[indexes]
        actions = np.array([np.array(POSSIBLE_MOVES == a, dtype=np.float32).flatten() for a in self.actions[indexes]])
        outputs = list() 
        for i in range(0, states.shape[0]):
            state = states[i]
            state_t1 = states_t1[i]
            action = actions[i]
            y = self.model.predict(state, action)
            next_q_values = self.model.predict(state_t1, action) # Ignores the action
            future_reward = DISCOUNT*(np.amax(next_q_values))
            y[np.where(action == 1)[0]] = rewards[i] + future_reward
            outputs.append(y)
        outputs = np.array(outputs)
        return (states, actions, outputs)

    def init_model(self):
        layer1_input = 128 * (12) * (2)
        self.model = tetris_theano.Model([
                tetris_theano.Conv2DLayer(32, 3, 3, 1, 10, 20),
                tetris_theano.Conv2DLayer(32, 3, 3, 32, 8, 18),
                tetris_theano.Conv2DLayer(64, 3, 3, 32, 6, 16),
                tetris_theano.Conv2DLayer(128, 2, 2, 64, 4, 14),
                tetris_theano.Conv2DLayer(128, 2, 2, 128, 3, 13),
                tetris_theano.Flatten(),
                tetris_theano.Split([tetris_theano.DenseLayer(layer1_input, 256),
                                     tetris_theano.DenseLayer(256, len(POSSIBLE_MOVES))],
                                    [tetris_theano.DenseLayer(layer1_input, 256),
                                     tetris_theano.DenseLayer(256, 1)],
                                    tetris_theano.ActionAdvantageMerge())
            ])
        self.model.compile()
