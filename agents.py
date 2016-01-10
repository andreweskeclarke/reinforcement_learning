import math
import random

class Agent:
    def __init__(self, actions, options={}):
        self.rgen = random.SystemRandom() # cryptographically secure, unlike random
        self.actions = actions
        self.avg_rewards = [0 for a in self.actions]
        self.n_observations = [0 for a in self.actions]
        self.epsilon = options['epsilon']
        self.action_history = list()

    def __choose_exploitative_action__(self):
        raise Exception('Not implemented!')

    def __choose_exploratory_action__(self):
        raise Exception('Not implemented!')

    def __should_exploit__(self):
        raise Exception('Not implemented!')

    def choose(self):
        if self.__should_exploit__():
            self.action_history.append(self.__choose_exploitative_action__())
        else:
            self.action_history.append(self.__choose_exploratory_action__())
        return self.action_history[-1]

    def update(self, reward, state=None):
        last_action = self.action_history[-1]
        avg = self.avg_rewards[last_action]
        self.n_observations[last_action] += 1
        self.avg_rewards[last_action] = avg + (reward - avg)/self.n_observations[last_action]

class EGreedyAgent(Agent):
    def __init__(self, actions, options={}):
        super(EGreedyAgent, self).__init__(actions, options)

    def __choose_exploitative_action__(self):
        return self.avg_rewards.index(max(self.avg_rewards))

    def __choose_exploratory_action__(self):
        return self.rgen.choice(self.actions)

    def __should_exploit__(self):
        return self.rgen.random() < (1 - self.epsilon)

class EGreedySoftmaxAgent(EGreedyAgent):
    def __choose_exploratory_action__(self):
        choice = self.rgen.random()
        cumulative_probability = 1
        softmax_denominator = sum([math.exp(a) for a in self.avg_rewards])
        for a in self.actions:
            softmax_a = math.exp(self.avg_rewards[a]) / softmax_denominator
            cumulative_probability = cumulative_probability - softmax_a
            if cumulative_probability <= choice:
                return a
        assert(False)
