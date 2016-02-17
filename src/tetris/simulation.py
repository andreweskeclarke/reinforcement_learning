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

    if args.watch:
        a = GreedyAgent()
        Tetris(a).play_visually()
    else:
        a = Agent()
        Tetris(a).play()

if __name__ == "__main__":
    main()
