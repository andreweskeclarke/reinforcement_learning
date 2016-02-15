import queue
import threading
import time
from tetris_game import *
from tetris_agent import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run a tetris simulation')
    parser.add_argument('--watch', action='store_true', help='Run the newest models')
    args = parser.parse_args()

    actions_q = queue.Queue()
    states_q = queue.Queue()
    print_q = queue.Queue()
    if args.watch:
        a = GreedyAgent(actions_q, states_q, print_q)
        a.start()
        Tetris(actions_q, states_q, print_q).play_visually()
    else:
        a = Agent(actions_q, states_q, print_q)
        a.start()
        Tetris(actions_q, states_q, print_q).play()

if __name__ == "__main__":
    main()
