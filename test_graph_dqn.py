import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
from dqn_agent import DQNAgent
from sampler import Sampler
from fixed_policies import ZeroActionPolicy
from fixed_policies import RandomPolicy
import sys

env = gym.make("CartPole-v0")

def q_network(states):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h1, W2) + b2
    return q

session = tf.Session()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

if sys.argv[1] == "zero_action":
    policy = ZeroActionPolicy()
elif sys.argv[1] == "random_action":
    policy = RandomPolicy()

sampler = Sampler(policy, env, discount=1)

def action_masker(array):
    masked_action = np.zeros((array.size, num_actions))
    masked_action[np.arange(array.size), array] = 1
    return masked_action

dqn_agent = DQNAgent(session,
                     optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     summary_writer=writer)

for _ in trange(2000):
    batch = sampler.collect_one_batch()
    masked_action = action_masker(batch["actions"])
    next_action_probs = policy.probs_for_next_action(batch["rewards"])
    batch["next_action_probs"] = next_action_probs
    batch["actions"] = masked_action
    dqn_agent.update_parameters(batch)
