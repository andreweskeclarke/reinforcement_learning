import ipdb
from agents.reinforcement_agent import *

EMPTY_ACTIONS = np.zeros(len(POSSIBLE_MOVES))
TARGET_NET_UPDATE_FREQUENCY = 30000
EGREEDY_EPSILON = 0.9
N_INITIAL_RANDOM_MOVES = 50000
BATCH_SIZE = 8

class QAgent(Agent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.target_model = self.model.copy()
        self.exploiting_turn = bool(random.getrandbits(1))
        self.total_moves_performed = 0
        self.total_games_played = 0
        self.recent_q_values = deque([], N_ROLLING_AVG)

    def choose_action(self, board):
        if self.__should_exploit(): 
            state = Board.copy_state(board, board.tetronimo)
            q_values = self.model.predict(state, EMPTY_ACTIONS)
            self.recent_q_values.append(np.max(q_values))
            return np.argmax(q_values)
        return random.choice(POSSIBLE_MOVES)

    def game_over(self, total_reward):
        self.total_games_played += 1

    def on_episode_end(self, reward, episode_length):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        sys.stdout.flush()

    def on_move_end(self):
        self.total_moves_performed += 1
        self.__experience_replay()
        self.__update_target_model()

    def __should_exploit(self):
        if self.total_moves_performed < N_INITIAL_RANDOM_MOVES:
            return False
        return random.random() < EGREEDY_EPSILON

    def __max_populated_buffer_index(self):
        return min(self.total_moves_performed, BUFFER_SIZE)

    def __experience_replay(self):
        indexes = np.random.randint(0, self.__max_populated_buffer_index(), BATCH_SIZE)
        states, actions, target_q_values = self.__training_data_for_indexes(indexes)
        self.model.train(states, actions, target_q_values, 1)

    def __update_target_model(self):
        if self.total_moves_performed % TARGET_NET_UPDATE_FREQUENCY == 0:
            print('Copy over the target network')
            self.target_model = self.model.copy()

    def __training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        actions = self.actions_by_index(indexes)
        rewards = self.rewards[indexes]
        states_t1 = self.states_t1[indexes]
        targets = list() 
        for i in range(0, states.shape[0]):
            action = actions[i]
            current_targets = self.model.predict(states[i], EMPTY_ACTIONS)
            next_q_values = self.model.predict(states_t1[i], EMPTY_ACTIONS)
            next_target_q_values = self.target_model.predict(states_t1[i], EMPTY_ACTIONS)
            # Double DQN
            current_targets[np.where(action == 1)] = \
                rewards[i] + DISCOUNT*(next_target_q_values[np.argmax(next_q_values)])
            targets.append(current_targets)
        targets = np.array(targets)
        return (states, actions, targets)


