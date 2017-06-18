import argparse

from agents.piece_prediction_agent import *
from agents.reinforcement_agent import *
from agents.state_value_agent import *
from agents.state_value_prediction_agent import *
from agents.tetris_agent import *
from agents.q_agent import *

from tetris_game import *

AGENTS = {
    'reinforcement': ReinforcementAgent,
    'random': RandomAgent,
    'piece_prediction': PiecePredictionAgent,
    'state_value_agent': StateValueAgent,
    'state_value_prediction_agent': StateValuePredictionAgent,
    'q_agent': QAgent,
}

def main():
    parser = argparse.ArgumentParser(description='Run a tetris simulation')
    parser.add_argument('--agent', help='Which agent to run', required=True)
    parser.add_argument('--model', help='Which network architecture to run')
    parser.add_argument('--state_model', help='Which network architecture to run')
    parser.add_argument('--value_model', help='Which network architecture to run')
    parser.add_argument('--watch', action='store_true', help='Run the newest models', required=False, default=False)
    parser.add_argument('--saved_model_file', help='The location of the saved model to reload')
    args = parser.parse_args()

    if args.agent == 'state_value_agent':
        agent = AGENTS[args.agent](args.state_model, args.value_model)
        Tetris(agent).play()
    else:
        agent = AGENTS[args.agent](args.model, saved_model_file=args.saved_model_file, watch=args.watch)
        Tetris(agent).play()

if __name__ == "__main__":
    main()
