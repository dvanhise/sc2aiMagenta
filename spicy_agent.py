from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.lib import units
import numpy as np
import tensorflow as tf
import math

from spicy_model import SpicyModel


class SpicyAgent:

    action_options = [
        actions.FUNCTIONS.no_op.id,
        actions.FUNCTIONS.select_point.id,
        actions.FUNCTIONS.select_rect.id,
        # actions.FUNCTIONS.select_control_group.id,
        actions.FUNCTIONS.select_unit.id,
        actions.FUNCTIONS.select_army.id,
        actions.FUNCTIONS.Attack_screen.id,
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

    # Available argument types and their offset in the arg tensor
    arg_options = {
        'screen': 0,
        'screen2': 2,
        'select_unit_id': 4,
        'select_add': 5,
        'select_unit_act': 6,
        'select_point_act': 7,
        # 'control_group_act': 6,
    }

    SCREEN_SIZE = 64
    SCREEN_DEPTH = 9  # Number of screen views to use

    ARG_COUNT = 8  # Size of arg tensor

    UNIT_LENGTH = 7  # Length of unit tensor

    LEARNING_RATE = .0001
    DISCOUNT_RATE = .9
    VESPENE_SCALING = 1.5

    def __init__(self, model=None):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        # (input, output) tuples for the current scenario
        self.storage = []

        if not model:
            self.model = SpicyModel(self.SCREEN_SIZE, self.SCREEN_DEPTH, len(self.action_options), self.ARG_COUNT)
        else:
            self.model = model

        # self.model.summary()  # Print model architecture

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.storage = []
        self.steps = 0
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        # On first step, center camera in map (maybe best in map editor?)
        if self.steps == 1:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            # return actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [32, 32])

        screens_input, selected_input, available_actions_input = self.build_inputs_from_obs(obs)

        # print('Input Shapes: %s -- %s -- %s' % (screens_input.shape, selected_input.shape, available_actions_input.shape))
        output_actions, output_args = self.model.call(screens_input, selected_input, available_actions_input)

        # Add gaussian noise to action tensor
        output_actions += np.random.normal(scale=.05, size=output_actions.shape)

        # Clip the values to 1
        #   I don't think I should need to do this with sigmoid activation functions, yet I get errors
        output_actions = np.clip(output_actions, -.000001, .9999999)
        output_args = np.clip(output_args, None, .9999999)

        # Turn everything back to numpy so not funny business happens
        output_args = np.array(output_args).T
        output_actions = np.array(output_actions).T

        # print('Output Shapes: %s -- %s' % (output_actions.shape, output_args.shape))

        # Filter out unavailable actions before choosing the best one
        # TODO: Do this the proper numpy way
        for ndx in range(len(output_actions)):
            if self.action_options[ndx] not in obs.observation['available_actions']:
                output_actions[ndx] = 0.0
        action_id = self.action_options[int(np.argmax(output_actions))]

        # Build action
        action_args = []
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name == 'screen':
                x = output_args[self.arg_options['screen']]
                y = output_args[self.arg_options['screen'] + 1]
                action_args.append([math.floor(x * self.SCREEN_SIZE), math.floor(y * self.SCREEN_SIZE)])
                if x > 1 or y > 1:
                    print('This is the problem: x=%f, y=%f' % (x, y))
            elif arg.name == 'screen2':
                x = output_args[self.arg_options['screen2']]
                y = output_args[self.arg_options['screen2'] + 1]
                action_args.append([math.floor(x * self.SCREEN_SIZE), math.floor(y * self.SCREEN_SIZE)])
            elif arg.name == 'select_unit_id':
                select_unit_id = output_args[self.arg_options['select_unit_id']]
                local_index = math.floor(select_unit_id * len(self.unit_options))
                action_args.append([self.unit_options[local_index]])
            elif arg.name == 'select_add':
                select_add = output_args[self.arg_options['select_add']]
                action_args.append([math.floor(select_add * 2)])
            elif arg.name in 'select_unit_act':
                select_unit_act = output_args[self.arg_options['select_unit_act']]
                action_args.append([math.floor(select_unit_act * 4)])
            elif arg.name == 'select_point_act':
                select_point_act = output_args[self.arg_options['select_point_act']]
                action_args.append([math.floor(select_point_act * 4)])
            elif arg.name == 'queued':
                action_args.append([0])
            else:
                raise('Unrecognized function argument type: %s' % arg.name)

        # TODO: Add LSTM for the action
        
        self.storage.append((screens_input, selected_input, available_actions_input, output_actions, output_args))
        print("Action: %d, Args: %s" % (action_id, action_args))
        return actions.FunctionCall(action_id, action_args)

    # agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
    def update(self, replay_buffer, discount_rate, learning_rate, action_counter):
        pass

    def build_inputs_from_obs(self, obs):
        screens_input = np.zeros((self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE), dtype=np.float32)
        for ndx, name in enumerate(['player_id', 'player_relative', 'unit_type', 'selected', 'unit_hit_points',
                                    'unit_hit_points_ratio', 'active', 'unit_density', 'unit_density_aa']):
            screens_input[ndx] = np.array(obs.observation['feature_screen'][name])
        screens_input = screens_input.T
        screens_input = np.reshape(screens_input, (1, self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE))

        selected_input = np.zeros((self.UNIT_LENGTH, 10), dtype=np.float32)
        multi_select = np.array(obs.observation['multi_select'], dtype=np.float32)[:10]
        selected_input[0:multi_select.shape[0], 0:multi_select.shape[1]] = multi_select
        selected_input = np.reshape(selected_input, (1, self.UNIT_LENGTH, 10))

        available_actions_input = np.zeros(len(self.action_options), dtype=np.float32)
        available_actions = obs.observation['available_actions']
        for ndx, act_id in enumerate(self.action_options):
            available_actions_input[ndx] = (1. if act_id in available_actions else 0.)
        available_actions_input = np.reshape(available_actions_input, (1, len(self.action_options)))

        return screens_input, selected_input, available_actions_input

    def calc_reward(self, obs, obs_prev):
        score = obs.observation['score_by_category']
        score_prev = obs_prev.observation['score_by_category']
        # Difference in killed minerals and vespene - diff in lost minerals and vespene since last state
        diff_value_lost = (score[1] - score_prev[1]) + self.VESPENE_SCALING*(score[2] - score_prev[2]) - \
                          (score[3] - score_prev[3]) + self.VESPENE_SCALING*(score[4] - score_prev[4])

        score = obs.observation['score_by_vital']
        score_prev = obs.observation['score_by_vital']
        # Damage taken - damage dealt since last state
        diff_damage_done = (score[0] - score_prev[0]) - (score[1] - score_prev[1])

        reward = diff_value_lost + .5*diff_damage_done
        print('Agent reward: %.3f' % reward)
        return reward
