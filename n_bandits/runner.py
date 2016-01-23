import numpy as np
import yaml
from math import floor
import random
import matplotlib.pyplot as plt
import argparse

from bandits import NBandits, Bandit
from agents import *

class Environment:
    def __init__(self, bandits):
        self.bandits = bandits

    def possible_actions(self):
        return range(0, len(self.bandits))

    def take(self, action):
        was_action_optimal = self.bandits.is_action_optimal(action)
        return self.bandits.take(action), was_action_optimal

class Sim:
    def __init__(self, n_runs, n_plays, bandits_creator, agent_creator):
        self.n_runs = float(n_runs)
        self.n_plays = n_plays
        self.create_bandits = bandits_creator
        self.create_agent = agent_creator

    def run(self):
        self.optimal_choice_rates = [0] * self.n_plays
        for run in range(0,int(self.n_runs)):
            bandits = self.create_bandits(None)
            env = Environment(bandits)
            agent = self.create_agent(env.possible_actions())
            for i in range(0,self.n_plays):
                action = agent.choose()
                reward, was_optimal = env.take(action)
                agent.update(reward)
                if was_optimal:
                    self.optimal_choice_rates[i] += (1/self.n_runs)


    def runningMean(self, x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    def plot(self, color):
        plt.plot(range(0,self.n_plays),
                self.runningMean(self.optimal_choice_rates, 25),
                color=color)
        
def main():
    print("Start sim")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('settings', help='Location of the settings yaml file for the desired simulation')
    args = parser.parse_args()

    with open(args.settings, 'r') as stream:
        settings = yaml.load(stream)

    n_bandits = settings['n_bandits']
    n_runs = settings['n_runs']
    n_plays_per_run = settings['n_plays_per_run']

    for experiment in settings['experiments']:
        print(experiment)
        simulation = Sim(n_runs, n_plays_per_run,
                lambda _: eval(experiment['env_class'])(n_bandits, options=experiment['env_options']),
                lambda actions: eval(experiment['agent_class'])(actions, options=experiment['options']))
        simulation.run()
        simulation.plot(experiment['color'])

    plt.axis([0,n_plays_per_run,0,1.05])
    plt.title("NBandits Reinforcement")
    plt.plot([0,n_plays_per_run],[0.9, 0.9], '--', color='g')
    plt.plot([0,n_plays_per_run],[0.95, 0.95], '--', color='b')
    plt.show()

if __name__ == "__main__":
    main()
