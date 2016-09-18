import time
import curses
import math
import random
from collections import deque
from offsets import *

TIME_PER_TICK = 0
TIME_BETWEEN_ROUNDS = 0.5
N_ROLLING_AVG = 500

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
              lambda x:x ]

POSSIBLE_MOVE_NAMES = [
    "ROTATE_LEFT",
    "ROTATE_RIGHT",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "MOVE_DOWN",
    "DO_NOTHING" ]

POSSIBLE_MOVES = np.array([
    ROTATE_LEFT,
    ROTATE_RIGHT,
    MOVE_RIGHT,
    MOVE_LEFT,
    MOVE_DOWN,
    DO_NOTHING ],np.int8)

class Tetromino:
    def __init__(self, offsets, x, y, rotation_index):
        self.offsets = offsets
        self.x = x
        self.y = y
        self.rotation_index = rotation_index

    def rotate_left(self):
        return Tetromino(self.offsets, self.x, self.y, (self.rotation_index - 1) % 4)

    def rotate_right(self):
        return Tetromino(self.offsets, self.x, self.y, (self.rotation_index + 1) % 4)

    def move_left(self):
        return Tetromino(self.offsets, self.x - 1, self.y, self.rotation_index)

    def move_right(self):
        return Tetromino(self.offsets, self.x + 1, self.y, self.rotation_index)

    def move_down(self):
        return Tetromino(self.offsets, self.x, self.y - 1, self.rotation_index)

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
        self.tetronimo = None
        self.current_height = 0

    def __clear_line__(self, y):
        self.board_array = np.concatenate((
                            np.delete(self.board_array, (y), axis=0), 
                            [[0 for x in range(0,self.width)]]), 
                            axis=0)
    
    def __freeze_tetronimo__(self):
        n_cleared_rows = 0
        for x, y, v in self.tetronimo.occupied_squares():
            self.current_height = max(self.current_height, y + 1) # 0 based
            if not self.is_out(x, y):
                self.board_array[y][x] = 1
            if all(self.board_array[y]): 
                self.__clear_line__(y)
                n_cleared_rows += 1
        return n_cleared_rows

    def is_occupied(self, x, y):
        return self.board_array[y][x] != 0

    def is_out(self, x, y):
        return 0 > y or y >= self.height or x >= self.width or 0 > x

    def can_place_piece(self, tetronimo):
        for s_x, s_y, _ in tetronimo.occupied_squares():
            if self.is_out(s_x, s_y) or self.is_occupied(s_x, s_y):
                return False
        return True

    def tetronimo_settled(self):
        return self.tetronimo and not self.can_place_piece(self.tetronimo.move_down())

    def start_tetronimo(self, tetronimo):
        tetronimo.x = self.starting_x
        tetronimo.y = self.starting_y
        self.tetronimo = tetronimo

    def set_tetronimo(self, tetronimo):
        self.tetronimo = tetronimo

    def tick(self):
        old_height = self.current_height
        points = 0
        n_cleared_rows = 0
        if self.tetronimo is not None:
            if self.tetronimo_settled():
                n_cleared_rows = self.__freeze_tetronimo__()
                if n_cleared_rows > 0:
                    points = [0, 20, 40, 60, 100][n_cleared_rows]
                elif self.current_height <= old_height:
                    points = 2
                else:
                    points = -2
        return points, n_cleared_rows

    @classmethod
    def copy_state(cls, board, tetronimo):
        copy = np.array(board.board_array, copy=True, ndmin=3)
        if tetronimo is not None:
            for x, y, value in tetronimo.occupied_squares():
                copy[0][y][x] = -1
        return copy

class Tetris:
    def __init__(self, agent):
        self.agent = agent
        self.reset_tetronimos()

    def play_visually(self):
        curses.wrapper(self.play)
   
    def tick(self, board):
        next_tetronimo = board.tetronimo.move_down()
        if board.can_place_piece(next_tetronimo):
            board.set_tetronimo(next_tetronimo)
        new_reward, lines_cleared = board.tick()
        self.n_cleared += lines_cleared
        return new_reward

    def play(self, screen=None):
        print('Begin playing!')
        if screen is not None:
            self.init_colors()
        running_scores = deque([], N_ROLLING_AVG)
        n_games = 0
        print('output: n_game, avg_score, avg_q_value, n_lines, loss, accuracy, training_runs, epsilon, n_pieces')
        while self.agent.should_continue():
            board = Board()
            continue_game = True
            self.reset_tetronimos()
            board.start_tetronimo(self.generate_tetronimo(board))
            game_start = time.time()
            ticks = 0
            reward = 0
            self.n_cleared = 0
            n_pieces = 0
            while continue_game and n_pieces < 50:
                n_pieces += 1
                continue_game, episode_reward = self.play_episode(board)
                reward += episode_reward

            running_scores.append(reward)
            n_games += 1
            self.agent.n_games = n_games
            game_size = self.agent.current_game_length
            self.agent.game_over(reward)
            if screen is not None:
                print_game_over(board, board.tetronimo, reward, screen)
            else:
                avg = (sum(running_scores)/float(len(running_scores)))
                avg_q_value = 0
                avg_loss = 0
                avg_accuracy = 0
                if hasattr(self.agent, 'recent_q_values') and len(self.agent.recent_q_values) > 0:
                    avg_q_value = sum(self.agent.recent_q_values) / float(len(self.agent.recent_q_values))
                if hasattr(self.agent, 'recent_losses') and len(self.agent.recent_losses) > 0:
                    avg_loss = self.agent.recent_losses[-1]
                    avg_accuracy = self.agent.recent_accuracies[-1]
                #print('output: n_game, avg_score, avg_q_value, n_lines, loss, accuracy')
                print('output: {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(n_games, reward, avg_q_value, self.n_cleared, avg_loss, avg_accuracy, 0, self.agent.epsilon(), n_pieces))

    def play_episode(self, board):
        episode_reward = 0
        episode_length = 0
        plays_since_tick_counter = 0
        while True:
            state_t0 = Board.copy_state(board, board.tetronimo)
            action = self.agent.choose_action(board)
            plays_since_tick_counter += 1
            episode_length += 1
            if action == MOVE_DOWN:
                while not board.tetronimo_settled():
                    board.set_tetronimo(board.tetronimo.move_down())
                episode_reward += self.tick(board)
            else:
                next_tetronimo = MOVES_MAP[action](board.tetronimo)
                if board.can_place_piece(next_tetronimo):
                    board.set_tetronimo(next_tetronimo)
                if plays_since_tick_counter >= 6:
                    episode_reward += self.tick(board)

            if board.tetronimo_settled():
                state_t1 = Board.copy_state(board, board.tetronimo)
                self.agent.handle(state_t0, action, -1, state_t1)
                self.agent.on_episode_end(episode_reward, episode_length)

                tetronimo = self.generate_tetronimo(board)
                if board.can_place_piece(tetronimo):
                    board.start_tetronimo(tetronimo)
                    return True, episode_reward
                else:
                    return False, episode_reward
            else:
                state_t1 = Board.copy_state(board, board.tetronimo)
                self.agent.handle(state_t0, action, -1, state_t1)

    def reset_tetronimos(self):
        self.tetronimos = [T, L, J, O, I, S, Z, T, L, J, O, I, S, Z] # Official rules
        random.shuffle(self.tetronimos)

    def generate_tetronimo(self, board):
        if len(self.tetronimos) == 0:
            self.reset_tetronimos()
        return Tetromino(self.tetronimos.pop(), board.starting_x, board.starting_y, 0)

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
    resting_state = Board.copy_state(board, tetronimo)
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
