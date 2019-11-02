from pysc2.lib import actions
from pysc2.lib import units
from pysc2.lib import features

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.losses as KL
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Concatenate, Reshape, Input, LSTM
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import Model

import math
import time

# from spicy_model import SpicyModel
from spicy_config import *


class SpicyAgent:

    action_options = [
        actions.FUNCTIONS.no_op.id,
        actions.FUNCTIONS.select_point.id,
        actions.FUNCTIONS.select_rect.id,
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

    arg_options = [
        'select_unit_id',
        'select_add',
        'select_unit_act',
        'select_point_act',
    ]

    SCREEN_SIZE = 64
    SCREEN_DEPTH = 9  # Number of screen views to use

    ARG_COUNT = 8  # Size of arg tensor

    UNIT_TENSOR_LENGTH = 3

    SELECT_SIZE = 3
    MAX_UNIT_SELECT = SCREEN_SIZE - len(action_options)

    VESPENE_SCALING = 1.5
    UNIT_HP_SCALE = 200  # 1600 by default

    def __init__(self, name='agent'):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.name = name
        self.obs_spec = None
        self.action_spec = None

        self.recorder = []

        self.model = self.build_model(self.SCREEN_SIZE,
                                      self.SCREEN_SIZE,
                                      self.SCREEN_DEPTH,
                                      self.UNIT_TENSOR_LENGTH,
                                      len(self.action_options))
        self.model.summary()
        self.opt = RMSprop(lr=LEARNING_RATE)
        # self.opt = SGD(lr=LEARNING_RATE)

        # Output placeholders
        # self.value_pl = K.placeholder(shape=(None,))
        # self.actual_value_pl = K.placeholder(shape=(None,))
        # self.advantage_pl = K.placeholder(shape=(None,))
        # self.reward_pl = K.placeholder(shape=(None,))
        # self.ns_policy_pl = K.placeholder(shape=(len(self.action_options), 3))
        # self.spatial_policy_pl = K.placeholder(shape=(self.SCREEN_SIZE, self.SCREEN_SIZE))

        # How to convert blizzard unit and building IDs to our subset of units
        def convert_unit_ids(x):
            if x in self.unit_options:
                return self.unit_options.index(x) / len(self.unit_options)
            return 1.
        self.convert_unit_ids = convert_unit_ids
        self.convert_unit_ids_vect = np.vectorize(convert_unit_ids)

        # How to convert 'player_relative' data
        def convert_player_ids(x):
            if x == 1:    # Self
                return 1.
            elif x == 4:  # Enemy
                return -1.
            else:         # Background usually
                return 0.
        self.convert_player_ids = convert_player_ids
        self.convert_player_ids_vect = np.vectorize(convert_player_ids)

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.recorder = []
        self.steps = 0
        self.episodes += 1

    # Uses the experience from recorded data to train model
    def train(self):
        t1 = time.time()
        discounted_reward = 0
        for state in reversed(self.recorder):
            if not state.done:
                discounted_reward = np.float32(state.reward + DISCOUNT_RATE * discounted_reward)
                self.train_one_step(state.inputs, discounted_reward, state.ns_action, state.s_action)

        print('Training round complete for %s, time taken: %.1f' % (self.name, time.time() - t1))

    def train_one_step(self, inputs, reward, action, screen_action):
        with tf.GradientTape() as tape:
            spatial_policy, ns_policy, value = self.model(inputs)
            ns_weights = ns_policy[:, :, 0]   # Only look at action probability part of ns_policy

            action_one_hot = K.one_hot(action, len(self.action_options))
            screen_action_one_hot = K.one_hot(screen_action, self.SCREEN_SIZE * self.SCREEN_SIZE)

            advantage = reward - value

            # entropy = K.sum(KL.categorical_crossentropy(ns_weights, ns_weights)) + \
            #           K.sum(KL.categorical_crossentropy(spatial_policy, spatial_policy))
            entropy = K.sum(ns_weights * K.log(ns_weights + 1e-10)) + \
                      K.sum(spatial_policy * K.log(spatial_policy + 1e-10))
            value_loss = KL.mean_squared_error(value, reward)
            policy_loss = KL.categorical_crossentropy(action_one_hot, ns_weights, from_logits=True) * \
                          KL.categorical_crossentropy(screen_action_one_hot, K.flatten(spatial_policy), from_logits=True) * \
                          advantage
            total_loss = policy_loss + value_loss - entropy * ENTROPY_RATE

        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # capped_gradients = [K.clip(grad, -1., 1.) for grad in gradients]
        # for grad in gradients:
        #     if np.any(np.isnan(grad)):
        #         raise ValueError('NaN found in gradient')

        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    def strip_reshape(self, arr):
        return np.reshape(arr, tuple(s for s in arr.shape if s > 1))

    # Takes a state and returns an action, also updates step information
    def step(self, obs, training=True):
        self.steps += 1
        # On first step, center camera in map (maybe best in map editor?)
        # if self.steps == 1:
        #     return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        #     return actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [32, 32])

        # Calculate reward of previous step and update it's state
        if self.steps != 1:
            reward = self.calc_reward(obs, self.recorder[-1].obs)
            self.recorder[-1].update(reward)

        screens_input, ns_input = self.build_inputs_from_obs(obs)
        spatial_action_policy, ns_action_policy, value = self.model([screens_input, ns_input])

        if np.any(np.isnan(ns_action_policy)) or np.any(np.isnan(spatial_action_policy)):
            raise ValueError('NaN found in output tensor')

        # Remove dimensions with length 1
        spatial_action_policy = self.strip_reshape(spatial_action_policy)
        ns_action_policy = self.strip_reshape(ns_action_policy)

        if training:
            screen_choice = np.random.choice(range(self.SCREEN_SIZE*self.SCREEN_SIZE),
                                             p=spatial_action_policy.flatten()/np.sum(spatial_action_policy))
        else:
            screen_choice = np.argmax(spatial_action_policy)

        screen_x = screen_choice // spatial_action_policy.shape[1]
        screen_y = screen_choice % spatial_action_policy.shape[1]

        # Clip the values to 1
        #   I don't think I should need to do this with sigmoid activation functions, yet I get errors
        action_weights = np.clip(ns_action_policy, None, .9999999)[:, 0]

        # Create new array with only weights of available actions
        action_probs = []
        available_actions = []
        for ndx, act_id in enumerate(self.action_options):
            if act_id in obs.observation['available_actions']:
                action_probs.append(action_weights[ndx])
                available_actions.append(act_id)

        # Compute softmax with unavailable actions removed, normalize again to get around numpy random choice bug
        action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        # action_probs = np.array(tf.math.softmax(action_probs))
        # action_probs /= action_probs.sum()

        if training:
            # Select from probability distribution
            choice = np.random.choice(range(len(action_probs)), p=action_probs)
        else:
            # Select highest probability
            choice = int(np.argmax(action_probs))

        action_args = ns_action_policy[choice, 1:]
        action_args = np.clip(action_args, 0., .999999)
        action_id = available_actions[choice]
        build_args = []
        args = iter(action_args)
        # Build action
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name == 'screen':
                build_args.append([screen_x, screen_y])
            # screen2 is only part of rect_select so use preset size based on screen1 to avoid the variable
            elif arg.name == 'screen2':
                build_args.append([(screen_x + self.SELECT_SIZE) % self.SCREEN_SIZE,
                                   (screen_y + self.SELECT_SIZE) % self.SCREEN_SIZE])
            elif arg.name == 'select_unit_id':
                select_unit_id = next(args)
                local_index = int(select_unit_id * len(self.unit_options))
                build_args.append([self.unit_options[local_index]])
            elif arg.name == 'select_add':
                select_add = next(args)
                build_args.append([int(select_add * 2)])
            elif arg.name == 'select_unit_act':
                select_unit_act = next(args)
                build_args.append([int(select_unit_act * 4)])
            elif arg.name == 'select_point_act':
                select_point_act = next(args)
                build_args.append([int(select_point_act * 4)])
            # Always set queued arg as false
            elif arg.name == 'queued':
                build_args.append([0])
            else:
                raise KeyError('Unrecognized function argument type: %s' % arg.name)

        # Checking validitiy of arguments
        if len(actions.FUNCTIONS[action_id].args) != len(build_args):
            raise ValueError('%d != %d' % (len(actions.FUNCTIONS[action_id].args), len(build_args)))

        self.recorder.append(
            State(obs,
                  (screens_input, ns_input),
                  (spatial_action_policy, ns_action_policy, value),
                  choice,
                  screen_choice)
        )
        # print('Coords: %d, %d' % (screen_x, screen_y))
        # print("Action: %s, Args: %s, %s" % (actions.FUNCTIONS[action_id].name, [a.name for a in actions.FUNCTIONS[action_id].args], build_args))
        return actions.FunctionCall(action_id, build_args)

    def build_inputs_from_obs(self, obs):
        # Subset of screens to use as inputs
        screens = ['player_relative', 'unit_type', 'selected', 'unit_hit_points',
                   'unit_hit_points_ratio', 'active', 'unit_density', 'unit_density_aa']

        screens_input = np.zeros((self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE), dtype=np.float32)
        for ndx, name in enumerate(screens):
            if name == 'player_relative':
                screens_input[ndx] = self.convert_player_ids_vect(np.array(obs.observation['feature_screen'][name]))
            elif name == 'unit_type':
                unit_types = np.array(obs.observation['feature_screen'][name])
                screens_input[ndx] = self.convert_unit_ids_vect(unit_types)
            elif name == 'unit_hit_points':
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]) / self.UNIT_HP_SCALE
            else:
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]) / getattr(features.SCREEN_FEATURES, name).scale

        # screens_input = screens_input.T
        screens_input = np.reshape(screens_input, (1, self.SCREEN_SIZE, self.SCREEN_SIZE, self.SCREEN_DEPTH))

        # Create screen-sized array to copy the non-spacial parts into
        ns_input = np.zeros((self.SCREEN_SIZE, 3))

        # Available actions
        act_count = len(self.action_options)
        act_input = np.zeros((act_count, self.UNIT_TENSOR_LENGTH))
        available_actions = obs.observation['available_actions']
        for ndx, act_id in enumerate(self.action_options):
            act_input[ndx, 0:self.UNIT_TENSOR_LENGTH] = (1. if act_id in available_actions else 0.)
        ns_input[0:act_count, 0:self.UNIT_TENSOR_LENGTH] = act_input

        # Normalizes the unit select tensor and removes fields
        def convert_select_tensor(x):
            return np.array([
                self.convert_unit_ids(x[0]),
                self.convert_player_ids(x[1]),
                x[2] / self.UNIT_HP_SCALE
            ], dtype=np.float32)

        # Selected units
        for ndx, unit in enumerate(obs.observation['multi_select']):
            ns_input[act_count + ndx:act_count + ndx + 1, 0:self.UNIT_TENSOR_LENGTH] = convert_select_tensor(unit)
        ns_input = np.reshape(ns_input, (1, self.SCREEN_SIZE, 3))

        if np.isin(np.NaN, ns_input) or np.isin(np.NaN, screens_input):
            raise ValueError('NaN found in input tensor')

        return screens_input, ns_input

    def calc_reward(self, obs, obs_prev):
        if obs.last():
            return 0.

        score = obs.observation['score_by_category'][2]
        score_prev = obs_prev.observation['score_by_category'][2]
        # Difference in killed minerals and vespene - diff in lost minerals and vespene since last state
        diff_value = (score[1] - score_prev[1]) + self.VESPENE_SCALING*(score[2] - score_prev[2]) - \
                     (score[3] - score_prev[3]) + self.VESPENE_SCALING*(score[4] - score_prev[4])

        score = obs.observation['score_by_vital'][2]
        score_prev = obs.observation['score_by_vital'][2]
        # Damage dealt - damage taken since last state
        diff_damage = (score[0] - score_prev[0]) - (score[1] - score_prev[1])

        reward = .01 * (diff_value + .5*diff_damage)
        # print('Change in reward: %.2f' % reward)
        return reward

    def build_model(self, screen_width, screen_height, screen_depth, ns_input_length, action_size):
        K.set_floatx('float32')

        # Inputs
        screen_input = Input(shape=(screen_width, screen_height, screen_depth), dtype='float32')
        ns_input = Input(shape=(screen_width, ns_input_length), dtype='float32')

        screen = Conv2D(screen_depth, 5, strides=1, padding='same', activation='relu')(screen_input)
        screen = Conv2D(2*screen_depth, 3, strides=1, padding='same', activation='relu')(screen)
        # screen = MaxPooling2D()(screen)

        ns = Dense(screen_height, use_bias=True, activation='relu')(ns_input)
        ns = Reshape((screen_width, screen_height, 1,))(ns)

        state = Concatenate(axis=3)([screen, ns])

        spacial_action_policy = Conv2D(1, 3, padding='same', activation='softmax')(state)
        spacial_action_policy = Reshape((screen_width, screen_height))(spacial_action_policy)

        state_ns = Conv2D(1, 5, strides=3, padding='valid', activation='relu')(state)
        state_ns = Flatten()(state_ns)
        state_ns = Dense(32, use_bias=True, activation='relu')(state_ns)

        ns_action_policy = Dense(action_size*3, use_bias=True)(state_ns)
        ns_action_policy = Reshape((action_size, 3))(ns_action_policy)
        ns_action_policy = LSTM(action_size*3, activation='sigmoid')(ns_action_policy)
        ns_action_policy = Reshape((action_size, 3))(ns_action_policy)

        value = Dense(1, use_bias=True)(state_ns)

        model = Model([screen_input, ns_input], [spacial_action_policy, ns_action_policy, value])

        return model


class State:
    def __init__(self, observation, inputs, outputs, ns_action, s_action):
        self.obs = observation
        self.inputs = inputs
        self.outputs = outputs
        self.next_obs = None
        self.reward = 0.
        self.ns_action = ns_action
        self.s_action = s_action
        self.done = True  # Assume done if state isn't updated

    def update(self, reward):
        self.reward = reward
        self.done = False
