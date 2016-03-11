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

BOARD_HEIGHT = 14
BOARD_WIDTH = 8

MOVES_MAP = [ lambda x:x.rotate_left(),
              lambda x:x.rotate_right(),
              lambda x:x.move_right(),
              lambda x:x.move_left(),
              lambda x:x.move_down(),
              lambda x:None ]

POSSIBLE_MOVES = np.array([
#    ROTATE_LEFT,
#    ROTATE_RIGHT,
    MOVE_RIGHT,
    MOVE_LEFT,
    MOVE_DOWN ],np.int8)
#    DO_NOTHING ],np.int8)

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
        self.starting_x = math.floor(width / 2)
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

    def should_add_tetronimo(self):
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
        next_state = np.array(self.board_array, copy=True, ndmin=3)
        if self.current_tetronimo is not None:
            self.current_tetronimo.move_down()
            next_state = np.array(self.board_array, copy=True, ndmin=3)
            if not self.current_tetronimo.can_move_down():
                n_cleared_rows = self.__freeze_tetronimo__()
                self.current_tetronimo = None
                if self.current_height > old_height:
                    points = points - (self.current_height - old_height)
                else:
                    points += 2
                points += [0, 15, 30, 60, 120][n_cleared_rows]
        return points, n_cleared_rows, next_state

class Tetris:
    def __init__(self, agent):
        self.agent = agent

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
            n_games += 1
            board = Board()
            play_on = True
            tetronimo = self.generate_tetronimo(board)
            board.add_tetronimo(tetronimo)
            game_start = time.time()
            ticks = 0
            reward = 0
            n_pieces = 1
            n_cleared = 0
            current_height = 0
            while play_on:
                old_reward = reward
                for i in range(0,4):
                    state_t0 = np.array(board.board_array, copy=True, ndmin=3)
                    merge_board_and_piece(state_t0, tetronimo)
                    action = self.agent.choose_action(state_t0)
                    MOVES_MAP[action](tetronimo)
                    state_t1 = np.array(board.board_array, copy=True, ndmin=3)
                    if not np.array_equal(state_t0, state_t1):
                        merge_board_and_piece(state_t1, tetronimo)
                        self.agent.handle(state_t0, action, reward - old_reward, state_t1)
                    if action == DO_NOTHING:
                        break
                    if action == MOVE_DOWN:
                        while not board.should_add_tetronimo():
                            new_reward, lines_cleared, state_t1 = board.tick()
                            reward += new_reward
                            n_cleared += lines_cleared

                new_reward, lines_cleared, state_t1 = board.tick()
                reward += new_reward
                n_cleared += lines_cleared

                if board.should_add_tetronimo():
                    n_pieces += 1
                    tetronimo = self.generate_tetronimo(board)
                    play_on = board.add_tetronimo(tetronimo)

                merge_board_and_piece(state_t1, tetronimo)
                self.agent.handle(state_t0, action, reward - old_reward, state_t1)

            running_scores.append(reward)
            self.agent.n_games = n_games
            if screen is not None:
                print_game_over(board, tetronimo, reward, screen)
            elif n_games > 50:
                avg = (sum(running_scores)/float(len(running_scores)))
                self.agent.avg_score = avg
                print('Average: {}, Game: {} pts, {} lines cleared, {} pieces ({} seconds, nth play: {})'.format(avg, reward, n_cleared, n_pieces, time.time() - game_start, n_games))

                if len(running_scores) >= N_ROLLING_AVG/10:
                    avg_q_value = sum(self.agent.recent_q_values) / float(len(self.agent.recent_q_values) + 0.01)
                    avg_loss = sum(self.agent.recent_losses) / float(len(self.agent.recent_losses) + 0.01)
                    avg_accuracy = sum(self.agent.recent_accuracies) / float(len(self.agent.recent_accuracies) + 0.01)
                    #print('output: n_game, avg_score, avg_q_value, n_lines, loss, accuracy')
                    print('output: {}, {}, {}, {}, {}, {}, {}, {}'.format(n_games, avg, avg_q_value, n_cleared, avg_loss, avg_accuracy, self.agent.training_runs, self.agent.epsilon()))


    def generate_tetronimo(self, board):
        # return Tetromino(board, random.choice([T, L, J, O, I, S, Z]))
        return Tetromino(board, random.choice([O]))

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
    resting_state = np.array(board.board_array, copy=True, ndmin=3)
    merge_board_and_piece(resting_state, tetronimo)
    tetris_print(resting_state, reward, screen)
    screen.addstr(board.height + 7, 0, 'GAME OVER!')
    screen.refresh()
    time.sleep(TIME_BETWEEN_ROUNDS)

def merge_board_and_piece(array, piece):
    if piece is None:
        return
    for x, y, value in piece.occupied_squares():
        array[0][y][x] = -1

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
