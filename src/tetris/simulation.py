import queue
import threading
import time
from tetris_game import *
from tetris_agent import *
import argparse


AGENTS = {
    'reinforcement': ReinforcementAgent,
    'random': RandomAgent,
    'piece_prediction': PiecePredictionAgent,
}

def main():
    parser = argparse.ArgumentParser(description='Run a tetris simulation')
    parser.add_argument('--agent', help='Which agent to run')
    parser.add_argument('--model', help='Which network architecture to run')
    parser.add_argument('--watch', action='store_true', help='Run the newest models')
    args = parser.parse_args()

    if args.watch:
        a = GreedyAgent()
        Tetris(a).play()
    else:
        agent = AGENTS[args.agent](args.model)
        Tetris(agent).play()

if __name__ == "__main__":
    main()
