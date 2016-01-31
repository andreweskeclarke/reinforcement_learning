import random
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
from agents import discrete_choice

# TODO helper methods - probably deserve a class
def card_point_value(card):
    if card < 11:
        return card
    elif 11 <= card < 14:
        return 10
    else:
        return 11

def score(cards):
    raw_score = sum([card_point_value(c) for c in cards])
    if raw_score > 21 and 14 in cards: # Demote the first Ace
        cards[cards.index(14)] = 1
        return score(cards)
    else:
        return raw_score

def has_soft_ace(cards):
    return 14 in cards


class Deck:
    def __init__(self):
        self.rgenerator = random.SystemRandom()
        self.shuffle()

    def shuffle(self):
        self.cards = [c for c in range(0,52)]
        self.dealt_cards = set()

    def deal(self):
        dealer = [self.hit(), self.hit()]
        player = [self.hit(), self.hit()]
        return dealer, player

    def hit(self):
        while True:
            card = self.rgenerator.choice(self.cards)
            if card not in self.dealt_cards:
                self.dealt_cards.add(card)
                # options are 2,3,4,5,6,7,8,9,10,J(11),Q(12),K(13),A(14)
                return (card % 13) + 2

class Episode:
    def __init__(self, deck, policy):
        self.deck = deck
        self.policy = policy

    def run(self):
        self.states = []
        self.blackjack = False
        self.deal()
        self.play_player()
        self.play_dealer()

    def reward(self):
        player_score = score(self.player_cards)
        dealer_score = score(self.dealer_cards)
        if player_score > 21:
            return -1
        if self.blackjack or dealer_score > 21 or player_score > dealer_score:
            return 1
        if player_score == dealer_score:
            return 0
        return -1

    def play_dealer(self):
        while score(self.dealer_cards) < 17 and score(self.player_cards) < 22:
            self.dealer_cards.append(self.deck.hit())

    def visible_dealer_points(self):
        card = self.dealer_cards[0]
        return score([card]) if card != 14 else 1 # Special code for an Ace

    def action(self):
        return self.policy.choose_action(score(self.player_cards),
                                         self.visible_dealer_points(),
                                         has_soft_ace(self.player_cards))

    def play_player(self):
        if score(self.player_cards) == 21 and score(self.dealer_cards) != 21:
            self.blackjack = True
        action = self.action()
        while action == Policy.HIT:
            self.update_states(action)
            self.player_cards.append(self.deck.hit())
            action = self.action()
        self.update_states(action)

    def update_states(self, action):
        s = score(self.player_cards)
        if s < 12 or s > 21:
            return # ignore
        self.states.append('{}-{}-{}-{}'.format(s, 
            self.visible_dealer_points(),
            has_soft_ace(self.player_cards),
            action))

    def deal(self):
        self.deck.shuffle()
        self.dealer_cards, self.player_cards = self.deck.deal()

class Policy:
    HIT = 'hit'
    STICK = 'stick'
    ACTIONS = [HIT, STICK]

    def __init__(self):
        self.action_values = {}
        self.rgen = random.SystemRandom() # cryptographically secure, unlike random

    def choose_action(self, score, dealer_score, soft_ace):
        if score < 12:
            return Policy.HIT
        if score >= 21:
            return Policy.STICK
        hit_key = '{}-{}-{}-hit'.format(score, dealer_score, soft_ace)
        stick_key = '{}-{}-{}-stick'.format(score, dealer_score, soft_ace)
        if hit_key in self.action_values and stick_key in self.action_values:
            hit_score = self.action_values[hit_key]['reward']
            stick_score = self.action_values[stick_key]['reward']
            greedy_action = Policy.HIT if hit_score > stick_score else Policy.STICK
            exploring_action = Policy.HIT if hit_score <= stick_score else Policy.STICK
            sample_size = min(self.action_values[hit_key]['n'], self.action_values[stick_key]['n'])
            if self.rgen.random() > (1.0 - (0.5 / math.log(sample_size + 1.0))):
                return exploring_action
            else:
                return greedy_action

        return self.rgen.choice(Policy.ACTIONS)

    def update(self, state_actions, reward):
        for a in state_actions:
            if a not in self.action_values:
                self.action_values[a] = {'n':0.0, 'reward':0.0}
            self.action_values[a]['n'] += 1.0
            n = self.action_values[a]['n']
            self.action_values[a]['reward'] = (((n-1.0) * self.action_values[a]['reward']) + float(reward)) / n

    def converged(self):
        for k, v in self.action_values.items():
            if v['n'] < 50:
                return False
        return len(self.action_values) > 1

def main():
    start = time.time()
    deck = Deck()
    policy = Policy()
    episode = Episode(deck, policy)
    rewards = []
    state_values = dict()
    while not policy.converged():
        episode.run()
        policy.update(episode.states, episode.reward())

    print('Took {} seconds'.format(time.time() - start))
    plot_3d(policy.action_values)

def plot_3d(values):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X = np.arange(1, 11, 1)
    Y = np.arange(12, 22, 1)
    X, Y = np.meshgrid(X, Y)
    Z1 = X * 0.0
    Z2 = X * 0.0
    for v in values:
        #'12-1-False': {'reward': 1.0, 'n': 2}
        player = int(v.split('-')[0])
        dealer = int(v.split('-')[1])

        # Use the value of the action under a greedy policy
        value = values[v.replace('hit', 'stick')]['reward']
        if player < 21:
            value_hit = values[v.replace('stick', 'hit')]['reward']
            value = max([value_hit, value])
        soft_ace = v.split('-')[2] == 'True'
        if soft_ace:
            Z1[player - 12][dealer - 1] = value
        else:
            Z2[player - 12][dealer - 1] = value

    ax1.plot_wireframe(X, Y, Z1, rstride=1, cstride=1)
    ax1.set_xlim3d(1, 10)
    ax1.set_ylim3d(12, 21)
    ax1.set_zlim3d(-1,1)
    ax1.set_title('Usable Ace')
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')

    ax2.plot_wireframe(X, Y, Z2, rstride=1, cstride=1)
    ax2.set_xlim3d(1, 10)
    ax2.set_ylim3d(12, 21)
    ax2.set_zlim3d(-1,1)
    ax2.set_title('No Usable Ace')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    
    plt.show()

if __name__ == "__main__":
    main()
