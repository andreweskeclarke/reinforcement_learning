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

BUFFER_SIZE = 7000
DISCOUNT = 0.5
BROADCAST_PORT = 50005

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
        self.recent_q_values = deque([], 10*N_REPLAYS_PER_ROUND)
        self.recent_accuracies = deque([], 5*N_ROLLING_AVG)
        self.recent_losses = deque([], 5*N_ROLLING_AVG)
        self.n_games = 0
        self.state_printer = StatePrinter()
        self.exploiting_turn = bool(random.getrandbits(1))
        self.rolled_over_buffer = False

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return 0

    def choose_action(self, state):
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
        for i, index in enumerate(indexes):
            self.rewards[index] = self.rewards[index] * total_reward_factor
        self.experience_replay()
        self.current_game_length = 0

    def backup_episode(self, reward):
        indexes = self.last_n_indexes(self.current_game_length)
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

    def experience_replay(self):
        if self.rolled_over_buffer:
            self.rolled_over_buffer = False
            mask = np.random.rand(BUFFER_SIZE) < 0.8
            X_train, Y_train = self.training_data_for_indexes(mask)
            self.model.train(X1_train, Y_train, 1)
            self.draw_output(X1_train, Y_train)

    def draw_output(self, X_train, Y_train):
        plt.axis('off')
        n_cols = 10
        n_rows = 3
        plt.figure(dpi=80, figsize=(n_rows,n_cols))
        for i in range(0,10):
            j = random.randint(0,len(X_train) - 1)
            frame = plt.subplot(n_cols,n_rows,3*i+1)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            x = np.reshape(X_train[j], (20,10))
            plt.pcolor(x, cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

            frame = plt.subplot(n_cols,n_rows,3*i+3)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            y = np.reshape(self.model.predict(X_train[j]), (20,10))
            plt.pcolor(y, cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)
            if i == 0:
                print(y)

        plt.tight_layout()
        plt.savefig('./board_predictions.png')
        plt.close()

    def input_size(self):
        return BOARD_HEIGHT*BOARD_WIDTH + len(POSSIBLE_MOVES)

    def output_size(self):
        return BOARD_HEIGHT*BOARD_WIDTH
        
    # def training_data_for_indexes(self, indexes):
    #     input = np.reshape(self.states_t0[indexes], (-1,200)) 
    #     action = np.array([np.array(POSSIBLE_MOVES == a, dtype=np.float32).flatten() for a in self.actions[indexes]])
    #     input = np.hstack((input, action))
    #     return (input,
    #             np.reshape(self.states_t1[indexes], (-1,self.output_size())))

    def training_data_for_indexes(self, indexes):
        return (np.reshape(self.states_t0[indexes], (-1,200)),
                np.reshape(self.states_t1[indexes], (-1,self.output_size())))

    def init_model(self):
        self.model = tetris_theano.Model([
                tetris_theano.Conv2DLayer(64, 4, 4),
                tetris_theano.Flatten(),
                tetris_theano.DenseLayer(64*17*7, 256),
                tetris_theano.DenseLayer(256, self.output_size())
            ])
        self.model.compile()
