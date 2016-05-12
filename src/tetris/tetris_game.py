import time
import queue
import curses
import math
import sys
import random
import threading
import numpy as np
from collections import deque
from termcolor import colored, cprint
from offsets import *

TIME_PER_TICK = 0
TIME_BETWEEN_ROUNDS = 0.5
N_ROLLING_AVG = 100

ROTATE_LEFT = 0
ROTATE_RIGHT = 1
MOVE_RIGHT = 2
MOVE_LEFT = 3
MOVE_DOWN = 4
DO_NOTHING = 5

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

MOVES_MAP = [ lambda x:x.rotate_left(),
              lambda x:x.rotate_right(),
              lambda x:x.move_right(),
              lambda x:x.move_left(),
              lambda x:x.move_down(),
              lambda x:None ]

POSSIBLE_MOVE_NAMES = [
    # "ROTATE_LEFT",
    # "ROTATE_RIGHT",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    # "MOVE_DOWN",
    "DO_NOTHING" ]

POSSIBLE_MOVES = np.array([
    # ROTATE_LEFT,
    # ROTATE_RIGHT,
    MOVE_RIGHT,
    MOVE_LEFT,
    # MOVE_DOWN,
    DO_NOTHING ],np.int8)

class Tetromino:
    def __init__(self, board, offsets):
        self.offsets = offsets
        self.board = board
        self.x = board.starting_x
        self.y = board.starting_y
        self.rotation_index = 0

    def rotate_left(self):
        return self.reposition(self.x, self.y, (self.rotation_index - 1) % 4)

    def rotate_right(self):
        return self.reposition(self.x, self.y, (self.rotation_index + 1) % 4)

    def move_left(self):
        return self.reposition(self.x - 1, self.y, self.rotation_index)

    def move_right(self):
        return self.reposition(self.x + 1, self.y, self.rotation_index)

    def move_down(self):
        return self.reposition(self.x, self.y - 1, self.rotation_index)

    def reposition(self, x, y, rotation_index):
        if self.can_reposition(x, y, rotation_index):
            self.x = x
            self.y = y
            self.rotation_index = rotation_index
            return True
        return False

    def can_move_down(self):
        return self.can_reposition(self.x, self.y - 1, self.rotation_index)

    def can_reposition(self, x, y, rotation_index):
        for s_x, s_y, _ in self.occupied_squares(x, y, rotation_index):
            if self.board.is_out(s_x, s_y) or self.board.is_occupied(s_x, s_y):
                return False
        return True

    def occupied_squares(self, x=None, y=None, rotation_index=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        rotation_index = self.rotation_index if rotation_index is None else rotation_index
        squares = []
        for index_y, row in enumerate(self.offsets[rotation_index]):
            for index_x, occupied in enumerate(row):
                if occupied:
                    # hacky math to figure out relative positions from the offsets
                    # offsets are centered on 1,1 in the 4x4 grid definition
                    squares.append([(3 - index_x) + x - 1, (3 - index_y) + y - 1, occupied]) 
        return squares

class Board:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT):
        self.height = height
        self.width = width
        self.starting_x = math.floor(width / 2) - 1
        self.starting_y = height - 2
        self.board_array = np.array([[0 for x in range(0, self.width)] for y in range(0, self.height)], np.int8)
        self.current_tetronimo = None
        self.current_height = 0

    def __clear_line__(self, y):
        self.board_array = np.concatenate((
                            np.delete(self.board_array, (y), axis=0), 
                            [[0 for x in range(0,self.width)]]), 
                            axis=0)
    
    def __freeze_tetronimo__(self):
        n_cleared_rows = 0
        for x, y, v in self.current_tetronimo.occupied_squares():
            self.current_height = max(self.current_height, y + 1) # 0 based
            if not self.is_out(x, y):
                self.board_array[y][x] = 1
            if all(self.board_array[y]): 
                self.__clear_line__(y)
                n_cleared_rows += 1
        return n_cleared_rows

    def current_tetronimo_settled(self):
        return self.current_tetronimo == None

    def add_tetronimo(self, tetronimo):
        self.current_tetronimo = tetronimo
        for x, y, v in self.current_tetronimo.occupied_squares():
            if self.board_array[y][x] > 0:
                return False
        return True

    def is_occupied(self, x, y):
        return self.board_array[y][x] != 0

    def is_out(self, x, y):
        return 0 > y or y >= self.height or x >= self.width or 0 > x

    def tick(self):
        old_height = self.current_height
        points = 0
        n_cleared_rows = 0
        if self.current_tetronimo is not None:
            self.current_tetronimo.move_down()
            if not self.current_tetronimo.can_move_down():
                n_cleared_rows = self.__freeze_tetronimo__()
                self.current_tetronimo = None
                if n_cleared_rows > 0:
                    points = [0, 20, 40, 60, 100][n_cleared_rows]
                elif self.current_height <= old_height:
                    points = 2
                else:
                    points = -2
        return points, n_cleared_rows

    def copy_board_state(self):
        copy = np.array(self.board_array, copy=True, ndmin=3)
        if self.current_tetronimo is not None:
            for x, y, value in self.current_tetronimo.occupied_squares():
                copy[0][y][x] = -2
        return copy

class Tetris:
    def __init__(self, agent):
        self.agent = agent
        self.tetronimos = []

    def play_visually(self):
        curses.wrapper(self.play)
   
    def play(self, screen=None):
        print('Begin playing!')
        if screen is not None:
            self.init_colors()
        running_scores = deque([], N_ROLLING_AVG)
        n_games = 0
        print('output: n_game, avg_score, avg_q_value, n_lines, loss, accuracy, training_runs, epsilon')
        while True:
            board = Board()
            continue_game = True
            self.reset_tetronimos()
            tetronimo = self.generate_tetronimo(board)
            board.add_tetronimo(tetronimo)
            game_start = time.time()
            ticks = 0
            reward = 0
            n_pieces = 1
            n_cleared = 0
            while continue_game:
                continue_episode = True
                episode_reward = 0
                plays_since_tick_counter = 0
                while continue_episode:
                    state_t0 = board.copy_board_state()
                    action = self.agent.choose_action(state_t0)
                    plays_since_tick_counter += 1
                    if action == DO_NOTHING:
                        plays_since_tick_counter = 0
                        new_reward, lines_cleared = board.tick()
                        reward += new_reward
                        episode_reward += new_reward
                        n_cleared += lines_cleared
                    elif action == MOVE_DOWN:
                        while not board.current_tetronimo_settled():
                            new_reward, lines_cleared = board.tick()
                            reward += new_reward
                            episode_reward += new_reward
                            n_cleared += lines_cleared
                    else:
                        MOVES_MAP[action](tetronimo)
                        # if plays_since_tick_counter >= 5:
                        #     new_reward, lines_cleared = board.tick()
                        #     reward += new_reward
                        #     episode_reward += new_reward
                        #     n_cleared += lines_cleared

                    continue_episode = not board.current_tetronimo_settled()
                    if continue_episode:
                        state_t1 = board.copy_board_state()
                        self.agent.handle(state_t0, action, episode_reward, state_t1)
                    else:
                        tetronimo = self.generate_tetronimo(board)
                        could_add_more = board.add_tetronimo(tetronimo)
                        state_t1 = board.copy_board_state()
                        self.agent.handle(state_t0, action, episode_reward, state_t1)
                        if not could_add_more:
                            episode_reward = -4
                        continue_game = could_add_more and n_pieces < 50

                self.agent.on_episode_end(episode_reward)
                n_pieces += 1

            running_scores.append(reward)
            self.agent.n_games = n_games
            game_size = self.agent.current_game_length
            self.agent.game_over(reward)
            if screen is not None:
                print_game_over(board, tetronimo, reward, screen)
            else:
                avg = (sum(running_scores)/float(len(running_scores)))
                n_games += 1
                avg_q_value = 0
                avg_loss = 0
                avg_accuracy = 0
                if len(self.agent.recent_q_values) > 0:
                    avg_q_value = self.agent.recent_q_values[-1]
                if len(self.agent.recent_losses) > 0:
                    avg_loss = self.agent.recent_losses[-1]
                    avg_accuracy = self.agent.recent_accuracies[-1]
                #print('output: n_game, avg_score, avg_q_value, n_lines, loss, accuracy')
                print('output: {}, {}, {}, {}, {}, {}, {}, {}'.format(n_games, reward, avg_q_value, n_cleared, avg_loss, avg_accuracy, 0, self.agent.epsilon()))


    def reset_tetronimos(self):
        self.tetronimos = [T, L, J, O, I, S, Z, T, L, J, O, I, S, Z] # Official rules
        # self.tetronimos = [O]

    def generate_tetronimo(self, board):
        if len(self.tetronimos) == 0:
            self.reset_tetronimos()
        random.shuffle(self.tetronimos)
        return Tetromino(board, self.tetronimos.pop())

    def init_colors(self):
        curses.start_color()
        curses.use_default_colors()
        colors = [ curses.COLOR_BLUE,
                   curses.COLOR_CYAN,
                   curses.COLOR_GREEN,
                   curses.COLOR_MAGENTA,
                   curses.COLOR_RED,
                   curses.COLOR_WHITE,
                   curses.COLOR_YELLOW ]
        curses.init_pair(0, curses.COLOR_WHITE, curses.COLOR_BLACK)
        for i, c in enumerate(colors):
            curses.init_pair(i + 1, c, curses.COLOR_BLACK)

def print_game_over(board, tetronimo, reward, screen):
    resting_state = board.copy_board_state()
    tetris_print(resting_state, reward, screen)
    screen.addstr(board.height + 7, 0, 'GAME OVER!')
    screen.refresh()
    time.sleep(TIME_BETWEEN_ROUNDS)

def tetris_print(array, reward, screen):
    curses.noecho()
    curses.curs_set(0)
    screen.erase()
    for y, row in reversed(list(enumerate(array[0]))):
        for x, value in enumerate(row):
            character = "\u2588" if value else "."
            color = curses.color_pair(value)
            screen.addch(len(array[0]) - y, 3*x, character, color)
            screen.addch(len(array[0]) - y, 3*x + 1, character, color)
    screen.addstr(len(array[0]) + 5, 0, 'Reward: {}'.format(reward))
    screen.refresh()
