import numpy as np


def get_discounted_rewards(rewards, discount_rate):
    discounted_reward = 0
    discounted_rewards = np.zeros(rewards.shape)
    # Calculate discounted rewards working backwards
    for ndx, reward in enumerate(np.flip(rewards)):
        discounted_reward = reward + discount_rate * discounted_reward
        discounted_rewards[ndx] = discounted_reward
    return np.flip(discounted_rewards)
