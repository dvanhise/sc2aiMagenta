import numpy as np
import matplotlib.pyplot as plt

from config import *


class Episode:
    def __init__(self):
        self.timesteps = []
        self.current_step = 0

        self.screen_input = np.zeros((MAX_TIMESTEPS, SCREEN_SIZE, SCREEN_SIZE, SCREEN_DEPTH))
        self.action_input = np.zeros((MAX_TIMESTEPS, len(ACTION_OPTIONS)))
        self.unit_input = np.zeros((MAX_TIMESTEPS, len(UNIT_OPTIONS)))

        self.value = np.zeros((MAX_TIMESTEPS,))
        self.nonspatial_output = np.zeros((MAX_TIMESTEPS, len(ACTION_OPTIONS)))
        self.spatial_output = np.zeros((MAX_TIMESTEPS, SCREEN_SIZE * SCREEN_SIZE))

        self.screen_used = np.zeros((MAX_TIMESTEPS,))
        self.nonspatial_action = np.zeros((MAX_TIMESTEPS,))   # Defaults to no-ops
        self.spatial_action = np.zeros((MAX_TIMESTEPS,))
        self.reward = np.zeros((MAX_TIMESTEPS,))

    # Updates the time-based input tensor with the new step inputs and returns the new input tensor
    # def input_step(self, inputs):
    #     self.screen_input[self.current_step] = inputs[0]
    #     self.action_input[self.current_step] = inputs[1]
    #     self.unit_input[self.current_step] = inputs[2]
    #
    #     return self.screen_input, self.action_input, self.unit_input

    def save_step(self, inputs, outputs, nonspatial_action, spatial_action, screen_used):
        self.screen_input[self.current_step] = inputs[0]
        self.action_input[self.current_step] = inputs[1]
        self.unit_input[self.current_step] = inputs[2]

        self.spatial_output[self.current_step] = outputs[0]
        self.nonspatial_output[self.current_step] = outputs[1]
        self.value[self.current_step] = outputs[2]

        self.screen_used[self.current_step] = 1. if screen_used else 0.
        self.nonspatial_action[self.current_step] = nonspatial_action
        self.spatial_action[self.current_step] = spatial_action

        self.current_step += 1

    def reward_last_step(self, reward):
        self.reward[self.current_step - 1] = reward

    # def get_input_for_step(self, step):
    #     screen = self.screen_input * np.array([
    #         np.ones((SCREEN_SIZE, SCREEN_SIZE, SCREEN_DEPTH))
    #         if step <= count
    #         else np.zeros((SCREEN_SIZE, SCREEN_SIZE, SCREEN_DEPTH))
    #         for count in range(step)
    #     ])
    #
    #     action = self.action_input * np.array([
    #         np.ones((len(ACTION_OPTIONS),))
    #         if step <= count
    #         else np.zeros((len(ACTION_OPTIONS),))
    #         for count in range(step)
    #     ])
    #
    #     select = self.unit_input * np.array([
    #         np.ones((len(UNIT_OPTIONS),))
    #         if step <= count
    #         else np.zeros((len(UNIT_OPTIONS),))
    #         for count in range(step)
    #     ])
    #
    #     return screen, action, select

    def make_action_plot(self, name=''):
        action_probs = np.mean(self.nonspatial_output[:self.current_step], axis=0)

        plt.clf()
        plt.plot(range(len(action_probs)), action_probs, 'bo', markersize=2)
        plt.ylabel('Mean Probability')
        plt.xlabel('Action Index')
        plt.title('%s Action Probabilities' % name)
        plt.yscale('log')
        plt.savefig('figures/action_fig_%s.png' % name)

    def make_screen_plot(self, name=''):
        screen_probs = np.mean(self.spatial_output[:self.current_step], axis=0)
        screen_probs = np.reshape(screen_probs, (SCREEN_SIZE, SCREEN_SIZE))

        plt.clf()
        plt.imshow(screen_probs.T, cmap='viridis')  # Transpose because SC2 switches x and y
        plt.colorbar()
        plt.title('%s Screen Location Probabilities' % name)
        plt.savefig('figures/screen_fig_%s.png' % name)

    def make_reward_plot(self, name=''):
        plt.clf()
        plt.plot(range(self.current_step), self.value[:self.current_step], 'bo', markersize=2)
        plt.plot(range(self.current_step), self.reward[:self.current_step], 'ro', markersize=2)
        # plt.plot(range(len(episode)), discounted_rewards, 'go', markersize=2)
        plt.ylabel('Value/Reward')
        plt.xlabel('Step')
        plt.title('%s Predicted Value and Reward' % name)
        plt.legend(['Value', 'Reward'])
        plt.savefig('figures/reward_fig_%s.png' % name)
