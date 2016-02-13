import pprint
import sys
import random
from termcolor import colored, cprint
from src.tetris.offsets import *

class Tetromino:
    def __init__(self, board, offsets):
        self.board = board
        self.offsets = offsets
        self.rotation_index = 0

    def center(self, x, y):
        self.x = x
        self.y = y
        
    def rotate(self, new_index):
        for s_x, s_y, _ in self.occupied_squares(rotation_index=new_index):
            if self.board.is_out(s_x, s_y) or self.board.is_occupied(s_x, s_y):
                return False
        self.rotation_index = new_index
        return True

    def rotate_left(self):
        return self.rotate((self.rotation_index - 1) % 4)

    def rotate_right(self):
        return self.rotate((self.rotation_index + 1) % 4)

    def move(self, x, y):
        for s_x, s_y, _ in self.occupied_squares(x=x, y=y):
            if self.board.is_out(s_x, s_y) or self.board.is_occupied(s_x, s_y):
                return False
        self.x = x
        self.y = y
        return True

    def move_left(self):
        return self.move(self.x - 1, self.y)

    def move_right(self):
        return self.move(self.x + 1, self.y)

    def move_down(self):
        val = self.move(self.x, self.y - 1)
        return val

    def occupied_squares(self, x=None, y=None, rotation_index=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        rotation_index = self.rotation_index if rotation_index is None else rotation_index
        squares = []
        for index_y, row in enumerate(self.offsets[self.rotation_index]):
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
        self.current_tetronimo.center(5, 20)
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
        while play_on:
            random.choice([lambda x:x.rotate_left(),
                           lambda x:x.rotate_right(),
                           lambda x:x.move_right(),
                           lambda x:x.move_left()])(tetronimo)
            reward += board.tick()
            if board.should_add_tetronimo():
                tetronimo = self.generate_tetronimo(board)
                play_on = board.add_tetronimo(tetronimo)

        self.print(board.board_array)
        print(reward)

    def print(self, array):
        # pprint.PrettyPrinter().pprint(board.board_array[::-1])
        colors = [ "red", "green", "yellow", "blue", "magenta", "cyan", "white" ]
        for row in reversed(array):
            for pos in row:
                if pos > 0:
                    sys.stdout.write(colored(u"\u2588", colors[pos - 1]))
                    sys.stdout.write(colored(u"\u2588", colors[pos - 1]))
                else:
                    sys.stdout.write(".")
                    sys.stdout.write(".")
            sys.stdout.write("\n")

    def generate_tetronimo(self, board):
        return Tetromino(board, random.choice([T, L, J, O, I, S, Z]))

