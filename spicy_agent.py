from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.lib import units
import numpy as np
import tensorflow as tf

from spicy_model import SpicyModel


class SpicyAgent:

    action_options = [
        actions.FUNCTIONS.no_op.id,
        actions.FUNCTIONS.select_point.id,
        actions.FUNCTIONS.select_rect.id,
        actions.FUNCTIONS.select_unit.id,
        actions.FUNCTIONS.select_army.id,
        actions.FUNCTIONS.Attack_screen.id,
        actions.FUNCTIONS.Attack_Attack_screen.id,
        actions.FUNCTIONS.Cancel_quick.id,
        actions.FUNCTIONS.HoldPosition_quick.id,
        actions.FUNCTIONS.Move_screen.id,
        actions.FUNCTIONS.Patrol_screen.id
    ]

    unit_options = [
        units.Terran.Marine,
        units.Terran.Marauder,
        units.Terran.Hellion
    ]

    REWARD_SCALING = 100

    # Map subset
    MIN_X = 21
    MAX_X = 43
    SCREEN_SIZE = 64

    MIN_Y = 21
    MAX_Y = 43

    SCREEN_DEPTH = 7
    MAX_ARGS = 4

    UNIT_LENGTH = 7

    def __init__(self, model=None):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        # (input, output) tuples for the current scenario
        self.storage = []

        if not model:
            self.model = SpicyModel(self.SCREEN_SIZE, self.SCREEN_DEPTH, len(self.action_options)+self.MAX_ARGS)
        else:
            self.model = model

        self.model.summary()  # Print model architecture

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.storage = []

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward

        # Build inputs
        screens_input = np.empty((self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE), dtype=np.float32)
        for ndx, name in enumerate(['player_id', 'player_relative', 'unit_type', 'selected', 'hit_points',
                                    'unit_density', 'unit_density_aa']):
            screens_input[ndx] = np.array(obs.observation['screen'][name])

        multi_select = np.array(obs.observation['multi_select'], dtype=np.float32)[:10]
        selected_input = np.pad(multi_select, (self.UNIT_LENGTH, 10))

        available_actions_input = np.empty(len(self.action_options), dtype=np.float32)
        available_actions = obs.observation['available_actions']
        for ndx, act_id in enumerate(self.action_options):
            available_actions_input[ndx] = (1.0 if act_id in available_actions else 0.0)

        output_actions, output_args = self.model.call(screens_input, selected_input, available_actions_input)

        # Filter out unavailable actions before choosing the best one
        # TODO: Do this the proper numpy way
        for ndx in range(len(output_actions)):
            if ndx not in available_actions:
                output_actions[ndx] = 0.0
        action_id = self.action_options[np.argmax(output_actions)]

        # Build action
        action_args = []
        iter_args = iter(output_args)
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name in ('screen', 'screen2'):
                action_args.append([next(iter_args) * (self.MAX_X-self.MIN_X) + self.MIN_X,
                                    next(iter_args) * (self.MAX_Y-self.MIN_Y) + self.MIN_Y])
            elif arg.name == 'minimap':
                raise ('Unused function argument type: %s' % arg.name)
            elif arg.name in ['select_add', 'queued', 'select_point_act']:
                action_args.append([0])
            else:
                raise('Unknown function argument type: %s' % arg.name)

        # TODO: Add LSTM for the action
        
        self.storage.append((screens_input, selected_input, available_actions_input, output_actions, output_args))
        return actions.FunctionCall(action_id, action_args)

    def train(self, reward):
        # TODO: do_a_train(inputs, outputs, reward)
        pass

    def calc_reward(self, obs, game_result):

        # TODO: Use these for better reward scores
        # score_by_category
        # killed_minerals = 1
        # killed_vespene = 2
        # lost_minerals = 3
        # lost_vespene = 4

        killed_value_units = obs.observation['score_cumulative'][5]
        total_damage_dealt = obs.observation['score_by_vital'][0]
        total_damage_taken = obs.observation['score_by_vital'][1]

        return game_result/2 + \
            .25 * np.tanh(killed_value_units / self.REWARD_SCALING) + \
            .25 * np.tanh(total_damage_dealt / self.REWARD_SCALING) - \
            .25 * np.tanh(total_damage_taken / self.REWARD_SCALING)
