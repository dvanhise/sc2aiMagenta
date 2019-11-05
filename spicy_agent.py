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

from spicy_config import *


class SpicyAgent:

    MAX_CONTROL_GROUPS = 3
    MAX_UNIT_SELECT = 3  # The highest index unit in a selection that can be selected individually

    SCREEN_SIZE = 64
    UNIT_TENSOR_LENGTH = 3
    SELECT_SIZE = 3

    VESPENE_SCALING = 1.5
    UNIT_HP_SCALE = 200  # 1600 by default

    # Feature screens to use as spatial input
    obs_screens = ['player_relative', 'unit_type', 'selected', 'unit_hit_points',
                   'unit_hit_points_ratio', 'active', 'unit_density', 'unit_density_aa']
    SCREEN_DEPTH = len(obs_screens)

    action_options = [
        {
            'id': actions.FUNCTIONS.no_op.id,
            'args': []
        },
        *({
            'id': actions.FUNCTIONS.select_point.id,
            'args': [act, 'screen']
        } for act in range(4)),
        {
            'id': actions.FUNCTIONS.select_rect.id,
            'args': [0, 'screen_rect']
        },
        {
            'id': actions.FUNCTIONS.select_rect.id,
            'args': [1, 'screen_rect']
        },
        # Every combination of control group action and group id
        # *({'id': actions.FUNCTIONS.select_control_group.id, 'args': [act, id]} for id in range(MAX_CONTROL_GROUPS) for act in range(3)),
        # Every combination of unit select options and unit to select
        *({'id': actions.FUNCTIONS.select_unit.id, 'args': [act, id]} for id in range(MAX_UNIT_SELECT) for act in range(4)),
        {
            'id': actions.FUNCTIONS.select_army.id,
            'args': [0]
        },
        {
            'id': actions.FUNCTIONS.select_army.id,
            'args': [1]
        },
        {
            'id': actions.FUNCTIONS.Attack_screen.id,
            'args': [0, 'screen']
        },
        # {
        #     'id': actions.FUNCTIONS.Cancel_quick.id,
        #     'args': [0]
        # },
        {
            'id': actions.FUNCTIONS.Move_screen.id,
            'args': [0, 'screen']
        },
        {
            'id': actions.FUNCTIONS.HoldPosition_quick.id,
            'args': [0]
        },
        {
            'id': actions.FUNCTIONS.Patrol_screen.id,
            'args': [0, 'screen']
        }
    ]

    # Master list of units agent "knows about"
    unit_options = [
        units.Terran.Marine,
        units.Terran.Marauder,
        units.Terran.Hellion
    ]

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
        self.opt = RMSprop(lr=LEARNING_RATE)

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

    def reset(self):
        self.recorder = []
        self.steps = 0
        self.episodes += 1

    # Train model with game data in self.recorder
    def train(self):
        # Calculate discounted rewards working backwards
        # Maybe also incorporate penalty for time taken and win/loss in final reward calculation
        discounted_reward = 0
        rewards = []
        for state in reversed(self.recorder):
            discounted_reward = state.reward + DISCOUNT_RATE * discounted_reward
            rewards.append(discounted_reward)
        rewards = list(reversed(rewards))

        return self.train_batch(
            self.strip_reshape(np.array([state.inputs[0] for state in self.recorder])),
            self.strip_reshape(np.array([state.inputs[1] for state in self.recorder])),
            np.array(rewards, dtype=np.float32),
            np.array([state.ns_action for state in self.recorder]),
            np.array([state.s_action for state in self.recorder])
        )

    def train_batch(self, input1, input2, reward, action, screen_action):
        _entropy = _policy_loss = _value_loss = 0.
        with tf.GradientTape() as tape:
            spatial_policy, ns_policy, value = self.model([input1, input2])
            ns_policy = K.softmax(ns_policy)

            action_one_hot = K.one_hot(action, len(self.action_options))
            screen_action_one_hot = K.one_hot(screen_action, self.SCREEN_SIZE * self.SCREEN_SIZE)

            advantage = reward - value

            entropy = K.sum(ns_policy * K.log(ns_policy + 1e-10)) + \
                      K.sum(spatial_policy * K.log(spatial_policy + 1e-10))
            value_loss = KL.mean_squared_error(value, reward)
            # Should that negative on policy loss be there?
            policy_loss = -(KL.categorical_crossentropy(action_one_hot, ns_policy, from_logits=True) *
                            KL.categorical_crossentropy(screen_action_one_hot, spatial_policy, from_logits=True) *
                            advantage)
            total_loss = policy_loss + value_loss - entropy * ENTROPY_RATE

            _entropy = K.mean(entropy)
            _policy_loss = K.mean(policy_loss)
            _value_loss = K.mean(value_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # capped_gradients = [K.clip(grad, -1., 1.) for grad in gradients]
        # for grad in gradients:
        #     if np.any(np.isnan(grad)):
        #         raise ValueError('NaN found in gradient')

        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(_value_loss), float(_policy_loss), float(_entropy)

    def strip_reshape(self, arr):
        return np.reshape(arr, tuple(s for s in arr.shape if s > 1))

    # Takes a state and returns an action, also updates step information
    def step(self, obs, training=True):
        self.steps += 1

        # Calculate reward of previous step and update its state
        if self.steps != 1:
            reward = self.calc_reward(obs, self.recorder[-1].obs)
            self.recorder[-1].update(reward)

        screens_input, ns_input = self.build_inputs_from_obs(obs)
        spatial_action_policy, ns_action_policy, value = self.model([screens_input, ns_input])
        # print('shape: %s  %s' % (spatial_action_policy.shape, ns_action_policy.shape))

        if np.any(np.isnan(ns_action_policy)) or np.any(np.isnan(spatial_action_policy)):
            raise ValueError('NaN found in output tensor')

        # Remove dimensions with length 1
        spatial_action_policy = self.strip_reshape(spatial_action_policy)
        ns_action_policy = self.strip_reshape(ns_action_policy)

        if training:
            screen_choice = np.random.choice(self.SCREEN_SIZE * self.SCREEN_SIZE,
                                             p=spatial_action_policy / np.sum(spatial_action_policy))
        else:
            screen_choice = np.argmax(spatial_action_policy)

        screen_x = screen_choice // self.SCREEN_SIZE
        screen_y = screen_choice % self.SCREEN_SIZE

        # Create new array with only weights of available actions
        action_probs = []
        available_actions = []
        for ndx, act_info in enumerate(self.action_options):
            if act_info['id'] in obs.observation['available_actions']:
                action_probs.append(ns_action_policy[ndx])
                available_actions.append((act_info, ndx))

        # Compute softmax with unavailable actions removed, normalize again to get around numpy random choice bug
        action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        # action_probs = np.array(tf.math.softmax(action_probs))
        # action_probs /= action_probs.sum()

        if training:
            # Select from probability distribution
            choice = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Select highest probability
            choice = int(np.argmax(action_probs))

        action, real_index = available_actions[choice]
        build_args = []
        # Build action
        for arg in action['args']:
            if arg == 'screen':
                build_args.append([screen_x, screen_y])
            elif arg == 'screen_rect':
                build_args.append([np.max([(screen_x - self.SELECT_SIZE), 0]),
                                   np.max([(screen_y - self.SELECT_SIZE), 0])])
                build_args.append([np.min([(screen_x + self.SELECT_SIZE), self.SCREEN_SIZE-1]),
                                   np.min([(screen_y + self.SELECT_SIZE), self.SCREEN_SIZE-1])])
            elif type(arg) is int:
                build_args.append([arg])
            else:
                raise KeyError('Unrecognized function argument: %s' % arg)

        self.recorder.append(
            State(obs,
                  (screens_input, ns_input),
                  (spatial_action_policy, ns_action_policy, value),
                  real_index,
                  screen_choice)
        )
        # print("Action: %s, Args: %s" % (actions.FUNCTIONS[action['id']].name, build_args))
        return actions.FunctionCall(action['id'], build_args)

    def build_inputs_from_obs(self, obs):
        screens_input = np.zeros((self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE), dtype=np.float32)
        # Transpose feature screens because spatial observations are (y,x) coordinates, everything else is (x,y)
        for ndx, name in enumerate(self.obs_screens):
            if name == 'player_relative':
                screens_input[ndx] = self.convert_player_ids_vect(np.array(obs.observation['feature_screen'][name]).T)
            elif name == 'unit_type':
                unit_types = np.array(obs.observation['feature_screen'][name]).T
                screens_input[ndx] = self.convert_unit_ids_vect(unit_types)
            elif name == 'unit_hit_points':
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]).T / self.UNIT_HP_SCALE
            else:
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]).T / getattr(features.SCREEN_FEATURES, name).scale

        screens_input = np.reshape(screens_input, (1, self.SCREEN_SIZE, self.SCREEN_SIZE, self.SCREEN_DEPTH))

        # Create screen-sized array to copy the non-spacial parts into
        ns_input = np.zeros((self.SCREEN_SIZE, 3))

        # Available actions
        act_count = len(self.action_options)
        act_input = np.zeros((act_count, self.UNIT_TENSOR_LENGTH))
        available_actions = obs.observation['available_actions']
        for ndx, act_info in enumerate(self.action_options):
            act_input[ndx, 0:self.UNIT_TENSOR_LENGTH] = (1. if act_info['id'] in available_actions else 0.)
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

        score = obs.observation['score_by_category']
        score_prev = obs_prev.observation['score_by_category']
        # Difference in army killed minerals and vespene cost minus diff in lost minerals and vespene since last state
        diff_value = (score[1][1] - score_prev[1][1]) + self.VESPENE_SCALING*(score[2][1] - score_prev[2][1]) - \
                     (score[3][1] - score_prev[3][1]) + self.VESPENE_SCALING*(score[4][1] - score_prev[4][1])

        score = obs.observation['score_by_vital']
        score_prev = obs_prev.observation['score_by_vital']
        # Difference in damage dealt minus damage taken since last state
        diff_damage = (score[0][0] - score_prev[0][0]) - (score[1][0] - score_prev[1][0])

        reward = .1 * (diff_value + .5*diff_damage)
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

        spatial_action_policy = Conv2D(1, 3, padding='same', activation='softmax')(state)
        # spatial_action_policy = Reshape((screen_width, screen_height))(spatial_action_policy)
        # spatial_action_policy = LSTM(256)(spatial_action_policy)
        spatial_action_policy = Flatten()(spatial_action_policy)

        state_ns = Conv2D(1, 5, strides=3, padding='valid', activation='relu')(state)
        state_ns = Flatten()(state_ns)
        state_ns = Dense(32, use_bias=False, activation='relu')(state_ns)

        ns_action_policy = Dense(action_size * 2, use_bias=True)(state_ns)
        ns_action_policy = Reshape((action_size * 2, 1))(ns_action_policy)
        ns_action_policy = LSTM(action_size)(ns_action_policy)
        ns_action_policy = Reshape((action_size,))(ns_action_policy)

        value = Dense(1, use_bias=True)(state_ns)

        model = Model([screen_input, ns_input], [spatial_action_policy, ns_action_policy, value])

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
