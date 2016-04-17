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

N_REPLAYS_PER_ROUND = 2000

# SARS - [State_0, Action, Reward, State_1]
STATE0_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
STATE1_INDEX = 3

BUFFER_SIZE = 25000
DISCOUNT = 0.85
BROADCAST_PORT = 50005
DESIRED_GAME_QUEUE_SIZE = 50

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
        self.save_requests = 0
        # Treat as a ring buffer
        self.max_pos = 0
        self.current_pos = 0
        self.n_plays = 0
        self.states_t0 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.actions = np.zeros([BUFFER_SIZE], dtype=np.int8)
        self.states_t1 = np.zeros((BUFFER_SIZE,1,BOARD_HEIGHT,BOARD_WIDTH), dtype=np.int8)
        self.rewards = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.interesting_games = deque([], 2*DESIRED_GAME_QUEUE_SIZE)
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
        self.n_warmups = 0
        self.exploiting_turn = bool(random.getrandbits(1))
        self.rolled_over_buffer = False

    def exploit(self):
        return random.random() < self.epsilon()

    def warming_up(self):
        return False

    def epsilon(self):
        return min((0.0 + 0.9 * (self.n_games)/10000), 0.9)

    def __log_choice__(self, state, vals):
        if random.random() < 0.0005:
            tbug('Some predicted values for a board. Trained against {} examples so far.'.format(self.training_runs))
            tbug(np.array(state, ndmin=4))
            tbug('[MOVE_RIGHT, MOVE_LEFT, MOVE_DOWN, DO_NOTHING]')
            tbug(vals)
            max_arg = np.argmax(vals)
            tbug('POSSIBLE_MOVES[{}] aka {} aka '.format(max_arg, POSSIBLE_MOVES[max_arg], POSSIBLE_MOVE_NAMES[max_arg]))

    def choose_action(self, state):
        if self.exploiting_turn:
            if not self.exploit():
                choice = random.choice(POSSIBLE_MOVES)
                while (choice == MOVE_DOWN or choice == DO_NOTHING) and random.random() < 0.9:
                    choice = random.choice(POSSIBLE_MOVES)
            
            else:
                vals = self.model.predict(np.array(state, ndmin=4), verbose=0)
                max_choice = np.argmax(vals)
                choice = POSSIBLE_MOVES[max_choice]
                self.recent_q_values.append(vals[0][max_choice])
        else:
            choice = random.choice(POSSIBLE_MOVES)
            while (choice == MOVE_DOWN or choice == DO_NOTHING) and random.random() < 0.8:
                choice = random.choice(POSSIBLE_MOVES)
        return choice
        
    def last_n_indexes(self, n):
        if n == 0:
            indexes = [(self.current_pos - 1) % BUFFER_SIZE]
        else:
            indexes = [x % BUFFER_SIZE for x in range(self.current_pos - 1, self.current_pos - 1 - n, -1)]
        return indexes

    def tick_forward(self):
        self.current_pos = (self.current_pos + 1) % BUFFER_SIZE
        self.n_plays += 1
        self.current_game_length += 1
        self.max_pos = min(self.max_pos + 1, BUFFER_SIZE)
        self.current_episode_length += 1
        if self.current_pos == 0: # Rolled over on indexes
            self.rolled_over_buffer = True
            self.last_avg_rewards = sum(self.rewards) / len(self.rewards)

    def keep_game(self, total_reward, indexes):
        n_games = len(self.interesting_games)
        if n_games <= self.interesting_games.maxlen and total_reward > 30:
            states, y = self.training_data_for_indexes(indexes)
            game_info = (total_reward, states, y)
            self.interesting_games.append(game_info)
        else: 
            for g in self.interesting_games:
                if g[0] < total_reward:
                    self.interesting_games.remove(g)
                    states, y = self.training_data_for_indexes(indexes)
                    game_info = (total_reward, states, y)
                    self.interesting_games.append(game_info)

    def game_over(self, total_reward):
        self.state_printer.send_to_websocket(self.states_t1[self.current_pos - 1])
        indexes = self.last_n_indexes(self.current_game_length)
        debug_s1 = ""
        debug_s2 = ""
        total_reward_factor = 1.0 + (float(total_reward) * 0.01)
        for i, index in enumerate(indexes):
            debug_s1 += str(self.rewards[index]) + ","
            self.rewards[index] = self.rewards[index] * total_reward_factor
            debug_s2 += str(self.rewards[index]) + ","
        tbug("Game:")
        tbug(debug_s1)
        tbug("---")
        tbug(debug_s2)
        tbug("")
        self.experience_replay()
        self.save()
        self.current_game_length = 0

    def backup_episode(self, reward):
        indexes = self.last_n_indexes(self.current_episode_length)
        debug_indexes = ""
        debug_s1 = ""
        debug_s2 = ""
        for i, index in enumerate(indexes):
            debug_indexes += str(index) + ","
            debug_s1 += str(self.rewards[index]) + ","
            self.rewards[index] += float(reward) * (DISCOUNT**i)
            debug_s2 += str(self.rewards[index]) + ","
        tbug(debug_indexes)
        tbug(debug_s1)
        tbug(debug_s2)
        tbug("")
        if self.current_episode_length > 1:
            tbug(self.states_t0[self.current_pos - 2])
            tbug(str(self.actions[self.current_pos - 2]) + " - " + POSSIBLE_MOVE_NAMES[self.actions[self.current_pos - 2]])
            tbug(self.states_t1[self.current_pos - 2])
            tbug('----->')
        tbug(self.states_t0[self.current_pos - 1])
        tbug(str(self.actions[self.current_pos - 1]) + " - " + POSSIBLE_MOVE_NAMES[self.actions[self.current_pos - 1]])
        tbug(self.states_t1[self.current_pos - 1])
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
        sys.stdout.flush()

    def save(self):
        self.save_requests += 1
        if self.save_requests == 100:
            self.save_requests = 0
            name = time.strftime("%m-%dT%H%M%S%Z")
            model_file = 'output/model_{}.json'.format(name)
            weights_file = 'output/weights_{}.h5'.format(name)
            open(model_file, 'w').write(self.model.to_json())
            self.model.save_weights(weights_file, overwrite=True)
            tbug('Saved: {} and {}'.format(model_file, weights_file))

    def experience_replay(self):
        if self.warming_up():
            return
        if self.rolled_over_buffer:
            print('TRAIN')
            self.rolled_over_buffer = False
            keep_training = True
            random_idxs = np.random.randint(0, min(self.n_plays, BUFFER_SIZE) - 1, min(self.n_plays, BUFFER_SIZE))
            states, y = self.training_data_for_indexes(random_idxs)
            loss_i1, accuracy = self.model.train_on_batch(states, y, accuracy=True)
            improvements = deque([], 3)
            while keep_training:
                random_idxs = np.random.randint(0, min(self.n_plays, BUFFER_SIZE) - 1, min(self.n_plays, BUFFER_SIZE))
                states, y = self.training_data_for_indexes(random_idxs)
                loss_i2, accuracy = self.model.train_on_batch(states, y, accuracy=True)
                improvements.append(abs((loss_i2 - loss_i1)) / float(loss_i1))
                print('{} - {} / {}  =  {}'.format(loss_i2, loss_i1, loss_i1, sum(improvements)/float(len(improvements))))
                keep_training = sum(improvements)/float(len(improvements)) > 0.001 and len(improvements) > 1
                sys.stdout.flush()
                loss_i1 = loss_i2

            self.training_runs += y.shape[0]
            self.recent_losses.append(loss_i2)
            self.recent_accuracies.append(accuracy)

    def training_data_for_indexes(self, indexes):
        y = self.model.predict(self.states_t0[indexes], verbose=0)
        future_rewards = DISCOUNT*(np.amax(self.model.predict(self.states_t1[indexes], verbose=0), axis=1))
        for i, a in enumerate(self.actions[indexes]):
            a_index = np.where(POSSIBLE_MOVES == a)[0]
            y[i][a_index] = self.rewards[indexes][i] + future_rewards[i]
        states = self.states_t0[indexes]
        return (states, y)

    def init_model(self):
    #     self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
    #     self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))
        self.model = Sequential()
       #  self.model.add(Convolution2D(64, 5, 5,
       #                                        activation='tanh',
       #                                        subsample=(2,2),
       #                                        init='he_uniform',
       #                                        input_shape=(1,BOARD_HEIGHT,BOARD_WIDTH)))
       #  self.model.add(Flatten())
       #  self.model.add(Dropout(0.5))

        self.model.add(Flatten(input_shape=(1,BOARD_HEIGHT,BOARD_WIDTH)))
        self.model.add(Dense(1024, activation='tanh', init='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='tanh', init='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='tanh', init='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='tanh', init='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='tanh', init='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(POSSIBLE_MOVES), activation='linear', init='he_uniform'))
        optim = SGD(lr=0.05, decay=0.0, momentum=0.5, nesterov=True)
        self.model.compile(loss='mae', optimizer=optim)


class GreedyAgent(Agent):
    def __init__(self):
        threading.Thread.__init__(self)
        self.init_model()
        self.state_printer = StatePrinter()

    def init_model(self):
        self.model = model_from_json(open(max(glob.iglob('output/model_*.json'), key=os.path.getctime)).read())
        self.model.load_weights(max(glob.iglob('output/weights_*.h5'), key=os.path.getctime))

    def game_over(self, total_reward):
        self.state_printer.send_to_websocket(self.states_t1[self.current_pos - 1])
        time.sleep(2)
        if self.n_games % 5 == 0:
            self.init_model()

    def on_episode_end(self, reward):
        pass

    def choose_action(self, state):
        time.sleep(1)
        self.state_printer.send_to_websocket(state)
        vals = self.model.predict(np.array(state, ndmin=4), verbose=0)
        print(state)
        print(POSSIBLE_MOVE_NAMES)
        print(vals)
        max_choice = np.argmax(vals)
        choice = POSSIBLE_MOVES[max_choice]
        self.recent_q_values.append(vals[0][max_choice])
        print(str(choice) + " -> " + POSSIBLE_MOVE_NAMES[choice])
        return choice

    def handle(self, s0, a, r, s1):
        pass
