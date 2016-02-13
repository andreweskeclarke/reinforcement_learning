import queue
import threading
import time

from tetris_game import *

class Agent(threading.Thread):
    def __init__(self, actions_q, states_q):
        threading.Thread.__init__(self)
        self.actions_q = actions_q
        self.states_q = states_q
        self.daemon = True

    def choose_action(self):
        return random.choice(POSSIBLE_MOVES)
        
    def handle_state(self, state):
        pass

    def run(self):
        while True:
            state = self.states_q.get()
            self.handle_state(state)
            self.actions_q.put(self.choose_action())
            time.sleep(TIME_PER_TICK / 3.0)

def main():
    actions_q = queue.Queue()
    states_q = queue.Queue()
    a = Agent(actions_q, states_q)
    a.start()
    Tetris(actions_q, states_q).play()

if __name__ == "__main__":
    main()
