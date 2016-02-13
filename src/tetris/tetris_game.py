import time
import math
import sys
import random
from termcolor import colored, cprint
from src.tetris.offsets import *

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
    def __init__(self, width=10, height=24):
        self.height = height
        self.width = width
        self.starting_x = math.floor(width / 2)
        self.starting_y = height - 2
        self.board_array = [[0 for x in range(0, self.width)] for y in range(0, self.height)] 
        self.current_tetronimo = None

    def __clear_line__(self, y):
        del self.board_array[y]
        self.board_array.append([0 for x in range(0, self.width)])
    
    def __freeze_tetronimo__(self):
        n_cleared_rows = 0
        for x, y, v in self.current_tetronimo.occupied_squares():
            if not self.is_out(x, y):
                self.board_array[y][x] = v
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
        if self.current_tetronimo is not None:
            n_cleared_rows = 0
            if not self.current_tetronimo.move_down():
                n_cleared_rows = self.__freeze_tetronimo__()
                self.current_tetronimo = None
            return n_cleared_rows

class Tetris:
    def __init__(self):
        pass

    def play(self):
        board = Board()
        reward = 0
        play_on = True
        tetronimo = self.generate_tetronimo(board)
        board.add_tetronimo(tetronimo)
        start_time = time.time()
        ticks = 0
        time_per_tick = 0.05
        while play_on:
            random.choice([lambda x:x.rotate_left(),
                           lambda x:x.rotate_right(),
                           lambda x:x.move_right(),
                           lambda x:x.move_left(),
                           lambda x:time.sleep(0.01)])(tetronimo)
            while time.time() - start_time > time_per_tick * (ticks + 1):
                ticks += 1
                reward += board.tick()
                tetris_print(board)
                print(reward)
                if board.should_add_tetronimo():
                    tetronimo = self.generate_tetronimo(board)
                    play_on = board.add_tetronimo(tetronimo)

        tetris_print(board)
        print(reward)

    def generate_tetronimo(self, board):
        return Tetromino(board, random.choice([T, L, J, O, I, S, Z]))

def tetris_print(board):
    squares_in_play = None
    if board.current_tetronimo is not None:
        squares_in_play = board.current_tetronimo.occupied_squares()
    colors = [ "red", "green", "yellow", "blue", "magenta", "cyan", "white" ]
    for y, row in reversed(list(enumerate(board.board_array))):
        for x, pos in enumerate(row):
            if pos > 0:
                sys.stdout.write(colored(u"\u2588", colors[pos - 1]))
                sys.stdout.write(colored(u"\u2588", colors[pos - 1]))
            elif squares_in_play and any([s[0] is x and s[1] is y for s in squares_in_play]):
                color_pos = [s[2] for s in squares_in_play if s[0] is x and s[1] is y][0] - 1
                sys.stdout.write(colored(u"\u2588", colors[color_pos]))
                sys.stdout.write(colored(u"\u2588", colors[color_pos]))
            else:
                sys.stdout.write(".")
                sys.stdout.write(".")
            sys.stdout.write(" ")
        sys.stdout.write("\n")
