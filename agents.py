import math
import random

class Agent:
    def __init__(self, actions, options={}):
        self.rgen = random.SystemRandom() # cryptographically secure, unlike random
        self.actions = actions
        self.last_action = None

    def __choose_exploitative_action__(self):
        raise Exception('Not implemented!')

    def __choose_exploratory_action__(self):
        raise Exception('Not implemented!')

    def __should_exploit__(self):
        raise Exception('Not implemented!')

    def __update__(self, reward, state=None):
        raise Exception('Not implemented!')

    def choose(self):
        if self.__should_exploit__():
            self.last_action = self.__choose_exploitative_action__()
        else:
            self.last_action = self.__choose_exploratory_action__()
        return self.last_action

    def update(self, reward, state=None):
        self.__update__(reward, state=None)

class EGreedyAgent(Agent):
    def __init__(self, actions, options={}):
        super(EGreedyAgent, self).__init__(actions, options)
        use_optimistic = 'optimistic' in options and options['optimistic']
        initial_reward = 5 if use_optimistic else 0
        self.avg_rewards = [0 for a in self.actions]
        self.n_observations = [0 for a in self.actions]
        self.epsilon = options['epsilon']

    def __choose_exploitative_action__(self):
        return self.avg_rewards.index(max(self.avg_rewards))

    def __choose_exploratory_action__(self):
        return self.rgen.choice(self.actions)

    def __should_exploit__(self):
        return self.rgen.random() < (1 - self.epsilon)

    def __update__(self, reward, state=None):
        last_action = self.last_action
        avg = self.avg_rewards[last_action]
        self.n_observations[last_action] += 1
        self.avg_rewards[last_action] = avg + (reward - avg)/self.n_observations[last_action]

def softmax_choice(rgen, actions, action_prefs):
    choice = rgen.random()
    cumulative_probability = 1
    softmax_denominator = sum([math.exp(p) for p in action_prefs])
    for a in actions:
        softmax_a = math.exp(action_prefs[a]) / softmax_denominator
        cumulative_probability = cumulative_probability - softmax_a
        if cumulative_probability <= choice:
            return a
    assert(False)

class EGreedySoftmaxAgent(EGreedyAgent):
    def __choose_exploratory_action__(self):
        return softmax_choice(self.rgen, self.actions, self.avg_rewards)

class ReinforcementComparisonAgent(Agent):
    def __init__(self, actions, options={}):
        super(ReinforcementComparisonAgent, self).__init__(actions, options)
        self.action_preferences = [0 for a in self.actions]
        self.alpha = options['alpha']
        self.beta = options['beta']
        self.reference_reward = 0
        self.last_action = None

    def __choose_exploitative_action__(self):
        raise 'Unreachable code was reached!'

    def __choose_exploratory_action__(self):
        return softmax_choice(self.rgen, self.actions, self.action_preferences)

    def __should_exploit__(self):
        return False

    def __update__(self, reward, state=None):
        old_pref = self.action_preferences[self.last_action]
        self.action_preferences[self.last_action] = old_pref + self.beta * (reward - self.reference_reward)
        self.reference_reward = self.reference_reward + self.alpha * (reward - self.reference_reward)



