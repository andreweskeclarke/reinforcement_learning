import queue
import threading
import time
from tetris_game import *
from agents.tetris_agent import *
from agents.reinforcement_agent import *
from agents.state_value_agent import *
from agents.piece_prediction_agent import *
import argparse


AGENTS = {
    'reinforcement': ReinforcementAgent,
    'random': RandomAgent,
    'piece_prediction': PiecePredictionAgent,
    'state_value_agent': StateValueAgent,
}

def main():
    parser = argparse.ArgumentParser(description='Run a tetris simulation')
    parser.add_argument('--agent', help='Which agent to run', required=True)
    parser.add_argument('--model', help='Which network architecture to run')
    parser.add_argument('--state_model', help='Which network architecture to run')
    parser.add_argument('--value_model', help='Which network architecture to run')
    parser.add_argument('--watch', action='store_true', help='Run the newest models', required=False, default=False)
    args = parser.parse_args()

    if args.watch:
        a = GreedyAgent()
        Tetris(a).play()
    elif args.agent == 'state_value_agent':
        agent = AGENTS[args.agent](args.state_model, args.value_model)
        Tetris(agent).play()
    else:
        agent = AGENTS[args.agent](args.model)
        Tetris(agent).play()

if __name__ == "__main__":
    main()
