import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tetris_theano
import models
import numpy as np
import random
import math
import socket
import sys
import os
import datetime
import time

from tetris_game import *
from tetris_game import BOARD_WIDTH, BOARD_HEIGHT
from collections import deque

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

N_REPLAYS_PER_ROUND = 2000
BUFFER_SIZE = 5000
DISCOUNT = 0.5
BROADCAST_PORT = 50005
DEBUG=False

def tbug(msg):
    if DEBUG:
        print(msg)


class WebSocketPrinter:
    def __init__(self, broadcast_port=BROADCAST_PORT):
        self.broadcast_port = broadcast_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', 0))

    def print(self, state):
        data = bytes("|".join([",".join(map(str, x)) for x in state[0]]) + "\n", 'ascii')
        self.sock.sendto(data, ("<broadcast>", self.broadcast_port))
        sys.stdout.flush()


class Agent():
    def __init__(self, model_name, max_training_batches=1):
        self.init_model(model_name)
        # Treat as a ring buffer
        self.current_pos = 0
        self.max_pos = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE], dtype=np.int8)
        self.states_t1 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.n_games = 0
        self.state_printer = WebSocketPrinter()
        self.current_game_length = 0
        self.current_episode_length = 0
        self.n_games = 0
        self.max_training_batches = max_training_batches
        self.n_training_batches = 0
        self.model_name = model_name

    def epsilon(self):
        return 1.0

    def choose_action(self, state):
        raise NotImplementedError()

    def game_over(self, total_reward):
        raise NotImplementedError()

    def on_episode_end(self, reward):
        raise NotImplementedError()

    def rolled_over(self):
        pass
        
    def last_n_indexes(self, n):
        if n == 0:
            indexes = [(self.current_pos - 1) % BUFFER_SIZE]
        else:
            indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 1 - n, -1)]
        return indexes

    def actions_by_index(self, indexes):
        return np.array([np.array(POSSIBLE_MOVES == a, dtype=np.float32).flatten() for a in self.actions[indexes]])

    def tick_forward(self):
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.current_game_length += 1
        self.current_episode_length += 1
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        if self.current_pos == 0: # Rolled over on indexes
            self.rolled_over()

    def handle(self, state0, action, reward, state1):
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = 0
        self.states_t1[self.current_pos] = state1
        self.tick_forward()

    def init_model(self, model_name):
        self.model = models.compile(model_name)

    def should_continue(self):
        return self.n_training_batches < self.max_training_batches

    def training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        actions = self.actions_by_index(indexes)
        return (states, actions, states_t1)


class RandomAgent(Agent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def choose_action(self, state):
        return random.choice(POSSIBLE_MOVES)
        
    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])

    def on_episode_end(self, reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
