from math import floor
import random
import matplotlib.pyplot as plt
import argparse

from bandits import NBandits, Bandit
from agents import Agent, EGreedyAgent

class Environment:
    def __init__(self, bandits):
        self.bandits = bandits

    def possible_actions(self):
        return range(0, len(self.bandits))

    def take(self, action):
        was_action_optimal = self.bandits.is_action_optimal(action)
        return self.bandits.take(action), was_action_optimal

class Sim:
    def __init__(self, n_bdts, n_runs, n_plays, eps):
        self.n_bdts = n_bdts
        self.n_runs = float(n_runs)
        self.n_plays = n_plays
        self.eps = eps

    def run(self):
        optimal_choice_rates = [0] * self.n_plays
        for run in range(0,int(self.n_runs)):
            bandits = NBandits(self.n_bdts)
            env = Environment(bandits)
            agent = EGreedyAgent(self.eps, env.possible_actions())
            for i in range(0,self.n_plays):
                action = agent.choose()
                reward, was_optimal = env.take(action)
                agent.update(reward)
                if was_optimal:
                    optimal_choice_rates[i] += (1/self.n_runs)
        return optimal_choice_rates
        
def main():
    parser = argparse.ArgumentParser(description='Sim')
    parser.add_argument('--bdts', type=int, help='Number of bandits to use in the simulation')
    parser.add_argument('--runs', type=int, help='Number of runs to make')
    parser.add_argument('--plays', type=int, help='Number of plays per run')
    parser.add_argument('--eps', type=float, help='Epsilon value for an e-greedy agent')
    args = parser.parse_args()

    print("Start sim")
    sim = Sim(args.bdts, args.runs, args.plays, args.eps)
    avgs = sim.run()
    plt.plot(range(0,args.plays),avgs)
    plt.axis([0,args.plays,0,1.05])
    plt.title("NBandits Reinforcement: {} bdts, {} runs, {} plays, {} epsilon".format(
            args.bdts, args.runs, args.plays, args.eps
        ))
    plt.show()

if __name__ == "__main__":
    main()
