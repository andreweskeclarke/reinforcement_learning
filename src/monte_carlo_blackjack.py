import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D

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
    def __init__(self, deck):
        self.deck = deck

    def run(self):
        self.states = []
        self.deal()
        self.play_player()
        self.play_dealer()

    def reward(self):
        _, player_points = self.points(self.player_cards)
        _, dealer_points = self.points(self.dealer_cards)
        if player_points == dealer_points or (player_points > 21 and dealer_points > 21):
            return 0
        elif player_points > dealer_points and player_points <= 21:
            return 1
        else:
            return -1


    def play_dealer(self):
        # Dealer policy: play up to 17
        cards = self.dealer_cards
        cards, points = self.points(cards)
        while points < 17:
            cards.append(self.deck.hit())
            cards, points = self.points(cards)
        self.dealer_cards = cards

    def dealer_points(self):
        card = self.dealer_cards[0]
        if card == 14:
            return 1
        else:
            return self.points([card])[1] # Only one dealer card visible

    def play_player(self):
        # User policy: hit until we reach 20, 21, or go bust
        cards = self.player_cards
        cards, points = self.points(cards)
        dealer_points = self.dealer_points()
        usable_ace = 14 in cards # Ace is usable if we can drop it down to 1
        self.states.append('{}-{}-{}'.format(points, dealer_points, usable_ace))
        while points < 20:
            cards.append(self.deck.hit())
            cards, points = self.points(cards)
            usable_ace = 14 in cards
            self.states.append('{}-{}-{}'.format(points, dealer_points, usable_ace))
        self.player_cards = cards

    def deal(self):
        self.deck.shuffle()
        self.dealer_cards, self.player_cards = self.deck.deal()

    def points(self, cards):
        raw_points = sum([self.card_point_value(c) for c in cards])
        if raw_points > 21 and 14 in cards: # Demote the first Ace
            cards[cards.index(14)] = 1
            return self.points(cards)
        else:
            return cards, raw_points

    def card_point_value(self, card):
        if card < 11:
            return card
        elif 11 <= card < 14:
            return 10
        else:
            return 11

def main():
    deck = Deck()
    episode = Episode(deck)
    rewards = []
    runs = 500000
    state_values = dict()
    for i in range(0,runs):
        episode.run()
        reward = episode.reward()
        for state in episode.states:
            if state not in state_values:
                state_values[state] = {'n':0.0, 'reward':0.0}
            state_values[state]['n'] += 1.0
            n = state_values[state]['n']
            state_values[state]['reward'] = (((n-1.0) * state_values[state]['reward']) + float(reward)) / n
    plot_3d(state_values)

def plot_3d(state_values):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    X = np.arange(2, 31, 1)
    Y = np.arange(1, 11, 1)
    X, Y = np.meshgrid(X, Y)
    Z = X * 0.0
    for state in state_values:
        #'12-1-False': {'reward': 1.0, 'n': 2}
        usable_ace = state.split('-')[2] == 'True'
        if usable_ace:
            player = int(state.split('-')[0])
            dealer = int(state.split('-')[1])
            value = state_values[state]['reward']
            Z[dealer - 1][player - 2] = value

    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()

if __name__ == "__main__":
    main()
