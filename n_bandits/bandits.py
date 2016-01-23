import random

class Bandit:
    def __init__(self):
        self.rgenerator = random.SystemRandom()
        self.mu = self.rgenerator.uniform(0,5)
        self.sigma = self.rgenerator.random() # 0-1

    def get_val(self):
        return self.rgenerator.gauss(self.mu, self.sigma)

class NBandits:

    def __init__(self, n, options={}):
        self.n = n
        self.bandits = list()
        for i in range(0,n):
            self.bandits.append(Bandit())

        mus = [b.mu for b in self.bandits]
        self.opt_action = mus.index(max(mus))

    def __len__(self):
        return len(self.bandits)

    def take(self, action):
        # action is index of bdt
        return self.bandits[action].get_val()

    def is_action_optimal(self, action):
        return action == self.opt_action
