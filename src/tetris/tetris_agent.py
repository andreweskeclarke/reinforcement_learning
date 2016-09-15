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

    def create_sample_images(self, indexes_mask):
        n_rows = np.sum(indexes_mask)
        n_cols = 3
        plt.axis('off')
        plt.figure(figsize=(n_cols,n_rows))
        X1_sample, X2_sample, Y_sample = self.training_data_for_indexes(indexes_mask)
        for index, _ in enumerate(X1_sample):
            x = X1_sample[index]
            action = X2_sample[index]
            y = Y_sample[index]
            y_pre = self.model.predict(x, action).reshape(BOARD_HEIGHT,BOARD_WIDTH)
            frame = plt.subplot(n_rows,n_cols,3*index+1)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(x.reshape(BOARD_HEIGHT,BOARD_WIDTH), cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

            frame = plt.subplot(n_rows,n_cols,3*index+2)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(y.reshape(BOARD_HEIGHT,BOARD_WIDTH), cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

            frame = plt.subplot(n_rows,n_cols,3*index+3)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(y_pre, cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directory, 'piece_predicton_{}.png'.format(
            datetime.datetime.today().strftime('%Y%m%dT%H%M%S'))))
        plt.close('all')

    def training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        actions = self.actions_by_index(indexes)
        return (states, actions, states_t1)


class ReinforcementAgent(Agent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.exploiting_turn = bool(random.getrandbits(1))

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return min([0.9, self.n_games / 50])

    def choose_action(self, state):
        if self.exploiting_turn: 
            return random.choice(POSSIBLE_MOVES)
        action = np.zeros(len(POSSIBLE_MOVES))
        q_values = self.model.predict(state, action)
        self.recent_q_values.append(np.max(q_values))
        return np.argmax(q_values)

    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        indexes = self.last_n_indexes(self.current_game_length)
        total_reward_factor = 1.0 + (float(total_reward) * 0.05)
        total_reward_factor = max(0.5, total_reward_factor)
        for i, index in enumerate(indexes):
            self.rewards[index] = self.rewards[index] * total_reward_factor
        self.experience_replay()
        self.current_game_length = 0

    def store_episode_information(self, reward):
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
        self.store_episode_information(reward)
        self.exploiting_turn = self.exploit()

    def experience_replay(self):
        sys.stdout.flush()
        if self.n_games > 0 and self.n_games % 5 == 0:
            mask = np.random.rand(min(0, BUFFER_SIZE)) < 0.3
            X1_train, X2_train, Y_train = self.training_data_for_indexes(mask)
            self.model.train(X1_train, X2_train, Y_train, 1)

    def training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        rewards = self.rewards[indexes]
        actions = self.actions_by_index(indexes)
        outputs = list() 
        for i in range(0, states.shape[0]):
            state = states[i]
            state_t1 = states_t1[i]
            action = np.array(actions[i], ndim=2)
            y = self.model.predict(state, action)
            next_q_values = self.model.predict(state_t1, action) # Ignores the action
            future_reward = DISCOUNT*(np.amax(next_q_values))
            y[np.where(action == 1)[0]] = rewards[i] + future_reward
            outputs.append(y)
        outputs = np.array(outputs)
        return (states, actions, outputs)


class StateValueAgent(ReinforcementAgent):
    def __init__(self, state_model_name, value_model_name):
        super().__init__(state_model_name)
        self.state_predictor = models.compile(state_model_name)
        self.value_predictor = models.compile(value_model_name)
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.exploiting_turn = bool(random.getrandbits(1))

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return 0.9

    def choose_action(self, state):
        if not self.exploiting_turn: 
            return random.choice(POSSIBLE_MOVES)
        action = np.zeros(len(POSSIBLE_MOVES))
        q_values = []
        for i, move in enumerate(POSSIBLE_MOVES):
            action = np.zeros(len(POSSIBLE_MOVES))
            action[i] = 1
            predicted_state = self.state_predictor.predict(state, action).reshape(1,1,BOARD_WIDTH*BOARD_HEIGHT)
            q_values.append(self.value_predictor.predict(predicted_state, action)[i])

        self.recent_q_values.append(np.max(q_values))
        return np.argmax(q_values)

    def update_final_rewards(self, total_reward):
        total_reward_factor = 1.0 + (float(total_reward) * 0.05)
        total_reward_factor = max(0.5, total_reward_factor)
        for index in self.last_n_indexes(self.current_game_length):
            self.rewards[index] = self.rewards[index] * total_reward_factor

    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        self.update_final_rewards(total_reward)
        self.experience_replay()
        self.current_game_length = 0

    def store_episode_information(self, reward):
        n_ineffective_actions = 0
        for i, index in enumerate(self.last_n_indexes(self.current_episode_length)):
            if np.array_equal(self.states_t0[index], self.states_t1[index]):
                n_ineffective_actions += 1
                self.rewards[index] = 0
            else:
                self.rewards[index] += float(reward) * (DISCOUNT**(i - n_ineffective_actions))
        self.current_episode_length = 0

    def on_episode_end(self, reward):
        self.store_episode_information(reward)
        self.exploiting_turn = self.exploit()

    def experience_replay(self):
        sys.stdout.flush()
        if self.n_games > 0 and self.n_games % 5 == 0:
            self.train(0.3)

    def rolled_over(self):
        for i in range(0,10):
            self.train(0.8)

    def train(self, percent=0.8):
        mask = np.random.rand(BUFFER_SIZE) < percent
        X_state, X_action, Y_state = self.state_training_data_for_indexes(mask)
        Y_state = Y_state.reshape((Y_state.shape[0],BOARD_HEIGHT*BOARD_WIDTH))
        state_cost = self.state_predictor.train(X_state, X_action, Y_state, 1)
        print('state mean error: {}'.format(math.sqrt(state_cost)))

        X_state, Y_value = self.value_training_data_for_indexes(mask)
        value_cost = self.value_predictor.train(X_state, np.array([]), Y_value, 1)
        print('value mean error: {}'.format(math.sqrt(value_cost)))

    def state_training_data_for_indexes(self, indexes):
        return (self.states_t0[indexes],
                self.actions_by_index(indexes),
                self.states_t1[indexes])

    def value_training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        discounted_rewards = list() 
        for i in range(0, states.shape[0]):
            state = states[i]
            action = np.zeros(len(POSSIBLE_MOVES))
            action[i] = 1
            state_t1 = self.state_predictor.predict(state, action)
            future_reward = self.value_predictor.predict(state_t1, action) # Ignores the action

            discounted_rewards.append(self.rewards[i] + DISCOUNT * future_reward)
        discounted_rewards = np.array(discounted_rewards)
        return (states, discounted_rewards)

class RandomAgent(Agent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def choose_action(self, state):
        return random.choice(POSSIBLE_MOVES)
        
    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])

    def on_episode_end(self, reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])


class PiecePredictionAgent(Agent):
    def __init__(self, model_name, max_training_batches=100):
        super().__init__(model_name, max_training_batches)
        self.directory = '/home/aclarke/tmp/tprediction/{}'.format(self.model_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with open(os.path.join(self.directory, 'stats.csv'), 'w+') as f:
            f.write('batch,cost\n')

    def choose_action(self, state):
        return random.choice(POSSIBLE_MOVES)
        
    def game_over(self, total_reward):
        pass

    def rolled_over(self):
        start = time.time()
        indexes = [random.randint(0,BUFFER_SIZE) for i in range(0,20)]
        indexes_mask = np.array([1 if i in indexes else 0 for i in range(0,BUFFER_SIZE)], dtype='bool')

        i = 0
        while True:
            print('Training batch {} ({} of {}) with {}...'.format(i,
                self.n_training_batches,
                self.max_training_batches,
                self.model_name))
            mask = np.random.rand(BUFFER_SIZE) < 1
            X1_train, X2_train, Y_train = self.training_data_for_indexes(mask)
            Y_train = Y_train.reshape((Y_train.shape[0],BOARD_HEIGHT*BOARD_WIDTH))
            cost = self.model.train(X1_train, X2_train, Y_train, 1)
            print('{} mean error'.format(math.sqrt(cost)))
            if i % 25 == 0:
                self.create_sample_images(indexes_mask)
            i += 1

        with open(os.path.join(self.directory, 'stats.csv'), 'a') as f:
            f.write('{},{}\n'.format(self.n_training_batches, cost))
        self.n_training_batches += 1

    def on_episode_end(self, reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        sys.stdout.flush()
