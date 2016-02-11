import pprint
import random
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
        for s_x, s_y in self.occupied_squares(rotation_index=new_index):
            if self.board.is_out(s_x, s_y) or self.board.is_occupied(s_x, s_y):
                return False
        self.rotation_index = new_index
        return True

    def rotate_left(self):
        return self.rotate((self.rotation_index - 1) % 4)

    def rotate_right(self):
        return self.rotate((self.rotation_index + 1) % 4)

    def move(self, x, y):
        for s_x, s_y in self.occupied_squares(x=x, y=y):
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
        return self.move(self.x, self.y - 1)

    def occupied_squares(self, x=None, y=None, rotation_index=None):
        x = x or self.x
        y = y or self.y
        rotation_index = rotation_index or self.rotation_index
        squares = []
        for index_x, row in enumerate(self.offsets[self.rotation_index]):
            for index_y, occupied in enumerate(row):
                if occupied:
                    # hacky math to figure out relative positions from the offsets
                    # offsets are centered on 1,1 in the 4x4 grid definition
                    squares.append([(3 - index_x) + x - 1, (3 - index_y) + y - 1]) 
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
        for x, y in self.current_tetronimo.occupied_squares():
            self.board_array[y][x] = 1
            if all(self.board_array[y]): 
                self.__clear_line__(y)
                n_cleared_rows += 1
        return n_cleared_rows

    def should_add_tetronimo(self):
        return self.current_tetronimo == None

    def add_tetronimo(self, tetronimo):
        self.current_tetronimo = tetronimo
        self.current_tetronimo.center(5, 20)
        for x, y in self.current_tetronimo.occupied_squares():
            if self.board_array[y][x] == 1:
                return False
        return True

    def is_occupied(self, x, y):
        return self.board_array[y][x] == 1

    def is_out(self, x, y):
        return 0 > y or y > self.height or x > self.width or 0 > x

    def tick(self):
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
        while play_on:
            if board.should_add_tetronimo():
                play_on = board.add_tetronimo(self.generate_tetronimo(board))
            else:
                reward += board.tick()
        pprint.PrettyPrinter().pprint(board.board_array[::-1])
        print(reward)

    def generate_tetronimo(self, board):
        return Tetromino(board, random.choice([T, L, J, O, I, S, Z]))

