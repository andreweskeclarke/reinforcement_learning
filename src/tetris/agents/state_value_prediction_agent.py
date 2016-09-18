from agents.reinforcement_agent import *

STATE_VALUE_AGENT_DISCOUNT = 0.8


class StateValuePredictionAgent(ReinforcementAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.value_predictor = models.compile(model_name)
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.exploiting_turn = bool(random.getrandbits(1))

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return 0.9

    def choose_action(self, board):
        if not self.exploiting_turn: 
            return random.choice(POSSIBLE_MOVES)
        possible_states = []
        default_state = Board.copy_state(board, board.tetronimo)
        for action in POSSIBLE_MOVES:
            if board.can_place_piece(MOVES_MAP[action](board.tetronimo)):
                possible_states.append(Board.copy_state(board, MOVES_MAP[action](board.tetronimo)))
            else:
                possible_states.append(default_state)
        action = np.argmax([self.value_predictor.predict(s, POSSIBLE_MOVES) for s in possible_states])
        return action

    def update_final_rewards(self, total_reward):
        total_reward_factor = 1.0 + (float(total_reward) * 0.05)
        total_reward_factor = max(0.5, total_reward_factor)
        for index in self.last_n_indexes(self.current_game_length):
            self.rewards[index] = self.rewards[index] * total_reward_factor

    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        self.update_final_rewards(total_reward)
        self.experience_replay()
        self.current_game_length = 0

    def store_episode_information(self, reward, length):
        n_ineffective_actions = 0
        for i, index in enumerate(self.last_n_indexes(length)):
            if np.array_equal(self.states_t0[index], self.states_t1[index]):
                n_ineffective_actions += 1
            else:
                self.rewards[index] += float(reward) * (STATE_VALUE_AGENT_DISCOUNT**(i - n_ineffective_actions))

    def on_episode_end(self, reward, episode_length):
        self.store_episode_information(reward, episode_length)
        self.exploiting_turn = self.exploit()

    def experience_replay(self):
        sys.stdout.flush()

    def rolled_over(self):
        indexes = [random.randint(0,BUFFER_SIZE) for i in range(0,20)]
        for i in range(0,2):
            self.train(0.8)

    def train(self, percent=0.8):
        mask = np.random.rand(BUFFER_SIZE) < percent
        X_state, X_action, Y_value = self.value_training_data_for_indexes(mask)
        value_cost = self.value_predictor.train(X_state, X_action, Y_value, 1)
        print('value mean error: {}'.format(math.sqrt(value_cost)))

    def value_training_data_for_indexes(self, indexes):
        return (self.states_t0[indexes],
                self.actions_by_index(indexes),
                np.array(self.rewards[indexes]).reshape(-1,1))
