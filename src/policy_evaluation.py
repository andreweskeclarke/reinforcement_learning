

class PolicySimulator:
    def __init__(self, grid_size=5):
        self.actions = ['UP', 'LEFT', 'RIGHT', 'DOWN']
        self.states = [s for s in range(0,grid_size)]

        self.state_values = [
                [0,10, 0, 5, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                ]
        self.policy = {
                'UP':0.25,
                'LEFT':0.25,
                'RIGHT':0.25,
                'DOWN':0.25,
                }
        self.rewards = {
                
                }

    def get_transition_probability(self, update_state, action, state):
        return 1

    def get_reward(self, update_state, action, state):
        if state == 


initial_value = state_values[updating_state]
for action in policy:
    prob_a = policy[action]
    for state in state_values:
        prob_state_given_a = get_transition_probability(updating_state, action, state)
        value = state_values[state]
        reward = get_reward(updating_state, action, state)
        state_values[updating_state] = prob_a * prob_s_given_a * (reward + discount * value)
