import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from pgq_agent import PGQAgent
from sampler import Sampler
from epsilon_greedy_policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer
import random
import sys

random.seed(42)
tf.set_random_seed(42)
np.random.seed(42)

env = gym.make("CartPole-v0")

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    """
    Use this wrapper for batch normalization
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, eps)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, eps)

def show_image(array):
    """show the final reward plot"""
    plt.figure()
    plt.plot(array)
    plt.title("Reward Progress")
    plt.xlabel("Iteration number")
    plt.ylabel("rewards")
    plt.grid()
    plt.show()

def pgq_network(states, is_training=False):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    z1 = tf.matmul(states, W1) + b1
    #bn1 = batch_norm_wrapper(z1, is_training)
    bn1 = z1
    h1 = tf.nn.relu(bn1)
    W_a = tf.get_variable("W_a", [20, num_actions],
                         initializer=tf.random_normal_initializer())
    b_a = tf.get_variable("b_a", [num_actions],
                         initializer=tf.constant_initializer(0))
    a = tf.matmul(h1, W_a) + b_a
    probs = tf.nn.softmax(a, dim=-1)
    probs = tf.stop_gradient(probs)
    a = a - tf.reduce_sum(tf.mul(a, probs), axis=1, keep_dims=True)

    W_v = tf.get_variable("W_v", [20, 1],
                        initializer=tf.random_normal_initializer())
    b_v = tf.get_variable("b_v", [1],
                        initializer=tf.constant_initializer(0))

    v = tf.matmul(h1, W_v) + b_v
    q = a + v
    return q

session = tf.Session()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
writer = tf.summary.FileWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

# Policy parameters for the exploration policy
epsilon = 0.9
target_update_rate = 0.1
pgq_agent = PGQAgent(session,
                     optimizer,
                     pgq_network,
                     state_dim,
                     num_actions,
                     target_update_rate=target_update_rate,
                     summary_writer=writer)
# Switch between greedy and exploratory policy
exploration_policy = EpsilonGreedyPolicy(pgq_agent, num_actions, epsilon)
# Always take greedy actions according to greedy policy
greedy_policy = EpsilonGreedyPolicy(pgq_agent, num_actions, 1.0)

# Sampler (collect trajectories using the present PGQ agent)
num_episodes = 10
training_sampler = Sampler(exploration_policy, env, num_episodes=num_episodes)
testing_sampler = Sampler(greedy_policy, env, num_episodes=5)

# Initializing ReplayBuffer
buffer_size = 100000
sample_size = 32
replay_buffer = ReplayBuffer(buffer_size)

def update_q_parameters(batch):
    pgq_agent.update_parameters(batch)


reward = []
for _ in trange(1000):
    batch = training_sampler.collect_one_batch()
    replay_buffer.add_batch(batch)
    if sample_size <= replay_buffer.num_items:
        random_batch = replay_buffer.sample_batch(sample_size) # replay buffer
        update_q_parameters(random_batch)
        testing_batch = testing_sampler.collect_one_batch()
        reward.append(testing_batch["rewards"].sum()/200)

show_image(reward)
