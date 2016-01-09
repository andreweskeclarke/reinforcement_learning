from math import floor
import random
import matplotlib.pyplot as plt
import argparse
import argparse


class Bandit:
    def __init__(self):
        self.rgenerator = random.SystemRandom()
        self.mu = self.rgenerator.uniform(0,5)
        self.sigma = self.rgenerator.random() # 0-1

    def get_val(self):
        return self.rgenerator.gauss(self.mu, self.sigma)

class NBandits:

    def __init__(self, n):
        self.n = n
        self.bandits = list()
        for i in range(0,n):
            self.bandits.append(Bandit())

        mus = [b.mu for b in self.bandits]
        self.opt_action = mus.index(max(mus))

    def val(self, action):
        # action is index of bdt
        return self.bandits[action].get_val()

    def __len__(self):
        return len(self.bandits)

    def was_act_opt(self, action):
        return action == self.opt_action

class Agent:
    def __init__(self, bdts, eps):
        self.bdts = bdts
        self.rgen = random.SystemRandom()
        self.actions = range(0,len(self.bdts))
        self.avg_rs = [5 for a in self.actions]
        self.n_obs = [0 for a in self.actions]
        self.eps = eps
        self.last_act_was_opt = False
    
    def tick(self):
        action = self.choose()
        r = self.take(action)
        self.update(action, r)

    def choose(self):
        if self.rgen.random() < (1 - self.eps):
            return self.avg_rs.index(max(self.avg_rs))
        return self.actions[int(floor(self.rgen.uniform(0,len(self.bdts))))]

    def take(self, action):
        return self.bdts.val(action) # action is index of bdt

    def update(self, action, r):
        avg = self.avg_rs[action]
        n_obs = self.n_obs[action] + 1
        self.avg_rs[action] = avg + (r - avg)/n_obs
        self.n_obs[action] += 1
        self.last_act_was_opt = self.bdts.was_act_opt(action)

class Sim:
    def __init__(self, n_bdts, n_runs, n_plays, eps):
        self.n_bdts = n_bdts
        self.n_runs = n_runs
        self.n_runs_f = float(n_runs)
        self.n_plays = n_plays
        self.eps = eps

    def run(self):
        avgs = [0] * self.n_plays
        for run in range(0,self.n_runs):
            agt = Agent(NBandits(self.n_bdts), self.eps)
            for i in range(0,self.n_plays):
                agt.tick()
                if agt.last_act_was_opt:
                    avgs[i] = avgs[i] + (1/self.n_runs_f)
        return avgs
        
def main():
    parser = argparse.ArgumentParser(description='Sim')
    parser.add_argument('--bdts', type=int, help='Number of bdts to use')
    parser.add_argument('--runs', type=int, help='Number of runs to make')
    parser.add_argument('--plays', type=int, help='Number of plays per run')
    parser.add_argument('--eps', type=float, help='Greedy Epsilon')
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
