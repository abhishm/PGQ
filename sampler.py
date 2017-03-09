import numpy as np

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 num_episodes=10,
                 max_step=200,
                 use_doubly_robust=False):
        self.policy = policy
        self.env = env
        self.num_episodes = num_episodes
        self.max_step = max_step
        self.use_doubly_robust = use_doubly_robust

    def collect_one_episode(self):
        state = self.env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in xrange(self.max_step):
            action = self.policy.sampleAction(state[np.newaxis,:])
            next_state, reward, done, _ = self.env.step(action)
            # appending the experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            # going to next state
            state = next_state
            if done:
                break
        return dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )

    def collect_one_batch(self):
        episodes = []
        for i_episode in xrange(self.num_episodes):
            episodes.append(self.collect_one_episode())
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        next_states = np.concatenate([episode["next_states"] for episode in episodes])
        dones = np.concatenate([episode["dones"] for episode in episodes])
        batch = dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )
        return batch

    def samples(self):
        return self.collect_one_batch()
