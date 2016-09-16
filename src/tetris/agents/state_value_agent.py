from agents.reinforcement_agent import *


class StateValueAgent(ReinforcementAgent):
    def __init__(self, state_model_name, value_model_name):
        super().__init__(state_model_name)
        self.state_predictor = models.compile(state_model_name)
        self.value_predictor = models.compile(value_model_name)
        self.recent_q_values = deque([], N_ROLLING_AVG)
        self.recent_accuracies = deque([], N_ROLLING_AVG)
        self.recent_losses = deque([], N_ROLLING_AVG)
        self.exploiting_turn = bool(random.getrandbits(1))

    def exploit(self):
        return random.random() < self.epsilon()

    def epsilon(self):
        return 0.9

    def choose_action(self, state):
        if not self.exploiting_turn: 
            return random.choice(POSSIBLE_MOVES)
        action = np.zeros(len(POSSIBLE_MOVES))
        q_values = []
        for i, move in enumerate(POSSIBLE_MOVES):
            action = np.zeros(len(POSSIBLE_MOVES))
            action[i] = 1
            predicted_state = self.state_predictor.predict(state, action).reshape(1,1,BOARD_WIDTH*BOARD_HEIGHT)
            q_values.append(self.value_predictor.predict(predicted_state, action))

        self.recent_q_values.append(np.max(q_values))
        return np.argmax(q_values)

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

    def store_episode_information(self, reward):
        n_ineffective_actions = 0
        for i, index in enumerate(self.last_n_indexes(self.current_episode_length)):
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

    def rolled_over(self):
        indexes = [random.randint(0,BUFFER_SIZE) for i in range(0,20)]
        image_mask = np.array([1 if i in indexes else 0 for i in range(0,BUFFER_SIZE)], dtype='bool')
        for i in range(0,5):
            self.train(0.8, image_mask)

    def train(self, percent=0.8, image_mask=None):
        mask = np.random.rand(BUFFER_SIZE) < percent
        X_state, X_action, Y_state = self.state_training_data_for_indexes(mask)
        Y_state = Y_state.reshape((Y_state.shape[0],BOARD_HEIGHT*BOARD_WIDTH))
        state_cost = self.state_predictor.train(X_state, X_action, Y_state, 1)
        print('state mean error: {}'.format(math.sqrt(state_cost)))

        X_state, X_action, Y_value = self.value_training_data_for_indexes(mask)
        value_cost = self.value_predictor.train(X_state, X_action, Y_value, 1)
        print('value mean error: {}'.format(math.sqrt(value_cost)))
        if image_mask is not None:
            self.create_sample_images(image_mask)

    def state_training_data_for_indexes(self, indexes):
        return (self.states_t0[indexes],
                self.actions_by_index(indexes),
                self.states_t1[indexes])

    def value_training_data_for_indexes(self, indexes):
        states = self.states_t0[indexes]
        states_t1 = self.states_t1[indexes]
        actions = self.actions_by_index(indexes)
        discounted_rewards = list() 
        for i in range(0, states.shape[0]):
            action = np.array([])
            future_reward = self.value_predictor.predict(states_t1[i], action) # Ignores the action
            discounted_rewards.append(self.rewards[i] + DISCOUNT * future_reward)
        discounted_rewards = np.array(discounted_rewards)
        return (states, actions, discounted_rewards)

    def create_sample_images(self, indexes_mask):
        n_rows = np.sum(indexes_mask)
        n_cols = 3
        plt.axis('off')
        plt.figure(figsize=(n_cols,n_rows))
        X_states, X_actions, Y_states = self.state_training_data_for_indexes(indexes_mask)
        for index, _ in enumerate(X_states):
            x = X_states[index]
            action = X_actions[index]
            y = Y_states[index]
            y_pre = self.state_predictor.predict(x, action).reshape(BOARD_HEIGHT,BOARD_WIDTH)
            frame = plt.subplot(n_rows,n_cols,3*index+1)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(x.reshape(BOARD_HEIGHT,BOARD_WIDTH), cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

            frame = plt.subplot(n_rows,n_cols,3*index+2)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(y.reshape(BOARD_HEIGHT,BOARD_WIDTH), cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

            frame = plt.subplot(n_rows,n_cols,3*index+3)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.pcolor(y_pre, cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)

        plt.tight_layout()
        path = os.path.join('/home/aclarke/tmp/state_value_agent', 'piece_predicton_{}.png'.format(
            datetime.datetime.today().strftime('%Y%m%dT%H%M%S')))
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path)
        plt.close('all')
