import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from sampler import Sampler
from epsilon_greedy_policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer
import sys

tf.set_random_seed(42)

env = gym.make("CartPole-v0")
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

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

def plot_bounds(array):
    """show the final reward plot"""
    lb, median, ub = array
    plt.fill_between(np.arange(len(median)), lb, ub, alpha=0.2)
    plt.plot(median)
    plt.title("Reward Progress")
    plt.xlabel("Iteration number")
    plt.ylabel("rewards")
    plt.grid()
    plt.show()

def q_network(states, is_training=False):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    z1 = tf.matmul(states, W1) + b1
    #bn1 = batch_norm_wrapper(z1, is_training)
    bn1 = z1
    h1 = tf.nn.relu(bn1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h1, W2) + b2
    return q

def build_graph():
    session = tf.Session()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    writer = tf.summary.FileWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

    # Policy parameters for the exploration policy
    epsilon = 0.9
    target_update_rate = 0.1
    dqn_agent = DQNAgent(session,
                         optimizer,
                         q_network,
                         state_dim,
                         num_actions,
                         target_update_rate=target_update_rate,
                         summary_writer=writer)
    # Switch between greedy and exploratory policy
    exploration_policy = EpsilonGreedyPolicy(dqn_agent, num_actions, epsilon)
    # Always take greedy actions according to greedy policy
    greedy_policy = EpsilonGreedyPolicy(dqn_agent, num_actions, 1.0)

    # Sampler (collect trajectories using the present dqn agent)
    num_episodes = 10
    training_sampler = Sampler(exploration_policy, env, num_episodes=num_episodes)
    testing_sampler = Sampler(greedy_policy, env, num_episodes=5)

    # Initializing ReplayBuffer
    buffer_size = 100000
    replay_buffer = ReplayBuffer(buffer_size)

    return dqn_agent, training_sampler, testing_sampler, replay_buffer



def compute_rewards():
    sample_size = 32
    rewards = []
    for _ in range(2):
        tf.reset_default_graph()
        dqn_agent, training_sampler, testing_sampler, replay_buffer = build_graph()
        reward = []
        for _ in trange(1000):
            batch = training_sampler.collect_one_batch()
            replay_buffer.add_batch(batch)
            if sample_size <= replay_buffer.num_items:
                random_batch = replay_buffer.sample_batch(sample_size) # replay buffer
                dqn_agent.update_parameters(batch)
                testing_batch = testing_sampler.collect_one_batch()
                reward.append(testing_batch["rewards"].sum()/200)
        rewards.append(reward)
    return rewards, dqn_agent, training_sampler, testing_sampler, replay_buffer

rewards, dqn_agent, training_sampler, testing_sampler, replay_buffer = compute_rewards()
plot_bounds(np.percentile(rewards, [10, 50, 90], axis=0))
