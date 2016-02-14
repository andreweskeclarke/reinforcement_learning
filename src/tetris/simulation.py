import queue
import threading
import time
from tetris_game import *
from tetris_agent import *


def main():
    actions_q = queue.Queue()
    states_q = queue.Queue()
    print_q = queue.Queue()
    a = Agent(actions_q, states_q, print_q)
    a.start()
    Tetris(actions_q, states_q, print_q).play()

if __name__ == "__main__":
    main()
