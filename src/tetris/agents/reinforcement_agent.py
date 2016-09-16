from agents.tetris_agent import *


class ReinforcementAgent(Agent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.exploiting_turn = bool(random.getrandbits(1))

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return min([0.9, self.n_games / 50])

    def choose_action(self, state):
        if self.exploiting_turn: 
            return random.choice(POSSIBLE_MOVES)
        action = np.zeros(len(POSSIBLE_MOVES))
        q_values = self.model.predict(state, action)
        self.recent_q_values.append(np.max(q_values))
        return np.argmax(q_values)

    def game_over(self, total_reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        indexes = self.last_n_indexes(self.current_game_length)
        total_reward_factor = 1.0 + (float(total_reward) * 0.05)
        total_reward_factor = max(0.5, total_reward_factor)
        for i, index in enumerate(indexes):
            self.rewards[index] = self.rewards[index] * total_reward_factor
        self.experience_replay()
        self.current_game_length = 0

    def store_episode_information(self, reward):
        indexes = self.last_n_indexes(self.current_episode_length)
        n_ineffective_actions = 0
        for i, index in enumerate(indexes):
            if np.array_equal(self.states_t0[index], self.states_t1[index]):
                n_ineffective_actions += 1
                self.rewards[index] = 0
            else:
                self.rewards[index] += float(reward) * (DISCOUNT**(i - n_ineffective_actions))
        self.current_episode_length = 0

    def on_episode_end(self, reward):
        self.store_episode_information(reward)
        self.exploiting_turn = self.exploit()

    def experience_replay(self):
        sys.stdout.flush()
        if self.n_games > 0 and self.n_games % 5 == 0:
            mask = np.random.rand(min(0, BUFFER_SIZE)) < 0.3
            X1_train, X2_train, Y_train = self.training_data_for_indexes(mask)
            self.model.train(X1_train, X2_train, Y_train, 1)

    def training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        rewards = self.rewards[indexes]
        actions = self.actions_by_index(indexes)
        outputs = list() 
        for i in range(0, states.shape[0]):
            state = states[i]
            state_t1 = states_t1[i]
            action = np.array(actions[i], ndim=2)
            y = self.model.predict(state, action)
            next_q_values = self.model.predict(state_t1, action) # Ignores the action
            future_reward = DISCOUNT*(np.amax(next_q_values))
            y[np.where(action == 1)[0]] = rewards[i] + future_reward
            outputs.append(y)
        outputs = np.array(outputs)
        return (states, actions, outputs)


