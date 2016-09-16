from agents.tetris_agent import *


class PiecePredictionAgent(Agent):
    def __init__(self, model_name, max_training_batches=100):
        super().__init__(model_name, max_training_batches)
        self.directory = '/home/aclarke/tmp/tprediction/{}'.format(self.model_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with open(os.path.join(self.directory, 'stats.csv'), 'w+') as f:
            f.write('batch,cost\n')

    def choose_action(self, state):
        return random.choice(POSSIBLE_MOVES)
        
    def game_over(self, total_reward):
        pass

    def rolled_over(self):
        start = time.time()
        indexes = [random.randint(0,BUFFER_SIZE) for i in range(0,20)]
        indexes_mask = np.array([1 if i in indexes else 0 for i in range(0,BUFFER_SIZE)], dtype='bool')

        i = 0
        while True:
            print('Training batch {} ({} of {}) with {}...'.format(i,
                self.n_training_batches,
                self.max_training_batches,
                self.model_name))
            mask = np.random.rand(BUFFER_SIZE) < 1
            X1_train, X2_train, Y_train = self.training_data_for_indexes(mask)
            Y_train = Y_train.reshape((Y_train.shape[0],BOARD_HEIGHT*BOARD_WIDTH))
            cost = self.model.train(X1_train, X2_train, Y_train, 1)
            print('{} mean error'.format(math.sqrt(cost)))
            if i % 2 == 0:
                self.create_sample_images(indexes_mask)
            i += 1

        with open(os.path.join(self.directory, 'stats.csv'), 'a') as f:
            f.write('{},{}\n'.format(self.n_training_batches, cost))
        self.n_training_batches += 1

    def on_episode_end(self, reward):
        self.state_printer.print(self.states_t1[self.current_pos - 1])
        sys.stdout.flush()

    def create_sample_images(self, indexes_mask):
        n_rows = np.sum(indexes_mask)
        n_cols = 3
        plt.axis('off')
        plt.figure(figsize=(n_cols,n_rows))
        X1_sample, X2_sample, Y_sample = self.training_data_for_indexes(indexes_mask)
        for index, _ in enumerate(X1_sample):
            x = X1_sample[index]
            action = X2_sample[index]
            y = Y_sample[index]
            y_pre = self.model.predict(x, action).reshape(BOARD_HEIGHT,BOARD_WIDTH)
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
        plt.savefig(os.path.join(self.directory, 'piece_predicton_{}.png'.format(
            datetime.datetime.today().strftime('%Y%m%dT%H%M%S'))))
        plt.close('all')
