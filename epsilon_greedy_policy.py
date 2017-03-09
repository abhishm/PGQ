import numpy as np

class EpsilonGreedyPolicy(object):
    def __init__(self, dqn_agent, num_actions, epsilon):
        self.dqn_agent = dqn_agent
        self.num_actions = num_actions
        self.epsilon = epsilon

    def sampleAction(self, state):
        # Implement a epsilon-greedy exploration policy
        action_values = self.dqn_agent.compute_all_q_values(state)[0]
        if np.random.random() <= self.epsilon:
            return np.argmax(action_values)
        else:
            return np.random.choice(self.num_actions)
