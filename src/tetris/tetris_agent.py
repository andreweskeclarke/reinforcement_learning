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
import socket
import sys
import os

N_REPLAYS_PER_ROUND = 25

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

BUFFER_SIZE = 500000 
DISCOUNT = 0.85
BROADCAST_PORT = 50005
DESIRED_EPISODE_QUEUE_SIZE = 100

class StatePrinter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', 0))

    def print(self, state):
        data = bytes("|".join([",".join(map(str, x)) for x in state[0]]) + "\n", 'ascii')
        self.sock.sendto(data, ("<broadcast>", BROADCAST_PORT))

class Agent():
    def __init__(self):
        self.init_model()
        self.save_requests = 0
        # Treat as a ring buffer
        self.max_pos = 0
        self.current_pos = 0
        self.n_plays = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE], dtype=np.int8)
        self.states_t1 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.interesting_episodes = deque([], 2*DESIRED_EPISODE_QUEUE_SIZE)
        self.current_episode_length = 0
        self.current_game_length = 0
        self.training_runs = 0
        self.recent_q_values = deque([], 10*N_REPLAYS_PER_ROUND)
        self.recent_accuracies = deque([], 5*N_ROLLING_AVG)
        self.recent_losses = deque([], 5*N_ROLLING_AVG)
        self.last_avg_rewards = 0
        self.n_games = 0
        self.state_printer = StatePrinter()
        self.avg_score = 0

    def exploit(self):
        return random.random() < self.epsilon()

    def warming_up(self):
        return len(self.interesting_episodes) < DESIRED_EPISODE_QUEUE_SIZE

    def epsilon(self):
        if self.warming_up():
            return 0.0
        elif self.avg_score < 10:
            return 0.7
        elif self.avg_score < 20:
            return 0.8
        elif self.avg_score < 40:
            return 0.9
        else:
            return 0.95

    def choose_action(self, state):
        if self.exploit():
            vals = self.model.predict(np.array(state, ndmin=4), verbose=0)
            if random.random() < 0.005:
                print('Some predicted values for a board. Trained against {} examples so far.'.format(self.training_runs))
                print(np.array(state, ndmin=4))
                print('[MOVE_RIGHT, MOVE_LEFT, MOVE_DOWN, DO_NOTHING]')
                print(vals)
                print(np.argmax(vals))
            choice = np.argmax(vals)
            self.recent_q_values.append(vals[0][choice])
        else:
            choice = random.choice(POSSIBLE_MOVES)
            if choice == MOVE_DOWN and self.warming_up():
                choice = random.choice(POSSIBLE_MOVES)
        return choice
        
    def last_n_indexes(self, n):
        if n == 0:
            indexes = [(self.current_pos - 1) % BUFFER_SIZE]
        else:
            indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 1 - n, -1)]
        return indexes

    def backup_episode(self, reward):
        indexes = self.last_n_indexes(self.current_episode_length)
        for i, index in enumerate(indexes):
            self.rewards[index] += float(reward) * (DISCOUNT ** (i+1))
        self.current_episode_length = 0
        return indexes

    def tick_forward(self):
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.n_plays += 1
        self.current_game_length += 1
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        self.current_episode_length += 1
        if self.current_pos == 0: # Rolled over on indexes
            self.last_avg_rewards = sum(self.rewards) / len(self.rewards)

    def game_over(self, total_reward):
        indexes = self.last_n_indexes(self.current_game_length)
        for i, index in enumerate(indexes):
            self.rewards[index] += float(total_reward) * 0.05
        self.current_game_length = 0

    def handle(self, state0, action, reward, state1):
        self.states_t0[self.current_pos] = state0
        self.actions[self.current_pos] = action
        self.rewards[self.current_pos] = reward
        self.states_t1[self.current_pos] = state1
        self.tick_forward()
        if reward is not 0:
            indexes = self.backup_episode(reward)
            if reward > 2:
                states, y = self.training_data_for_indexes(indexes)
                self.interesting_episodes.append((np.copy(states), np.copy(y)))
        if not self.warming_up():
            self.state_printer.print(state1)
            self.experience_replay()
            self.save()
        sys.stdout.flush()

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
        random_idxs = np.random.randint(0, min(self.n_plays, BUFFER_SIZE), N_REPLAYS_PER_ROUND)
        self.train_on_indexes(random_idxs)
        if bool(random.getrandbits(1)) and bool(random.getrandbits(1)):
            sample_idxs = np.random.randint(0, len(self.interesting_episodes), N_REPLAYS_PER_ROUND)
            for s in sample_idxs:
                episode = self.interesting_episodes[s]
                i = random.randint(0, len(episode[0]) - 1)
                states = np.array(episode[0][i], ndmin=4)
                y = np.array(episode[1][i], ndmin=2)
                self.train_on(states, y, False)

    def train_on_indexes(self, indexes, keep_results=True):
        states, y = self.training_data_for_indexes(indexes)
        self.train_on(self.states_t0[indexes], y, keep_results)

    def training_data_for_indexes(self, indexes):
        y = self.model.predict(self.states_t0[indexes], verbose=0)
        future_rewards = DISCOUNT*(np.amax(self.model.predict(self.states_t1[indexes], verbose=0), axis=1))
        for i, a in enumerate(self.actions[indexes]):
            a_index = np.where(POSSIBLE_MOVES == a)[0]
            y[i][a_index] = self.rewards[indexes][i] + future_rewards[i]
        states = self.states_t0[indexes]
        return (states, y)

    def train_on(self, states, y, keep_results=True):
        loss, accuracy = self.model.train_on_batch(states, y, accuracy=True)
        if keep_results:
            self.training_runs += y.shape[0]
            self.recent_losses.append(loss)
            self.recent_accuracies.append(accuracy)

    def init_model(self):
    #     self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
    #     self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))
         self.model = Sequential()
         self.model.add(Convolution2D(64, 3, 3, 
                             activation='tanh', 
                             subsample=(1,1),
                             init='uniform',
                             input_shape=(1,BOARD_HEIGHT,BOARD_WIDTH)))
         self.model.add(Flatten())
         self.model.add(Dropout(0.5))
         self.model.add(Dense(256, activation='tanh', init='uniform'))
         self.model.add(Dropout(0.5))
         self.model.add(Dense(256, activation='tanh', init='uniform'))
         self.model.add(Dropout(0.5))
         self.model.add(Dense(len(POSSIBLE_MOVES), activation='linear', init='he_uniform'))
         optim = SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=True)
         self.model.compile(loss='mae', optimizer=optim)

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
        # return random.choice(POSSIBLE_MOVES)
        return np.argmax(self.model.predict(np.array(state, ndmin=4), batch_size=1, verbose=0))

    def handle(self, s0, a, r, s1):
        pass
