from pysc2.lib import features

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Concatenate, Reshape, Input, LSTM, \
    BatchNormalization, Activation, Lambda, Dropout, ConvLSTM2D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import Model

from config import *
from episode import Episode
from utils import get_discounted_rewards


class MainAgent:
    def __init__(self, name='agent', reward_weights=None):
        self.reward = 0
        self.episode = 0
        self.name = name

        # Default reward weights
        self.reward_weights = {
            'enemy_killed_value': 1,
            'friendly_killed_value': 1,
            'killed_value': 1,
            'damage_taken': 1,
            'damage_given': 1,
            'damage': 1,
            'outcome': 1,
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)

        self.last_obs = None
        self.recorder = []

        self.model = self.build_model(SCREEN_SIZE, SCREEN_SIZE, SCREEN_DEPTH, UNIT_TENSOR_LENGTH, len(ACTION_OPTIONS))
        self.opt = RMSprop(lr=LEARNING_RATE)

        # How to convert blizzard unit and building IDs to our subset of units
        def convert_unit_ids(x):
            if x in UNIT_OPTIONS:
                return (UNIT_OPTIONS.index(x) + 1.) / len(UNIT_OPTIONS)
            return 0.
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
        self.recorder = [Episode()]
        self.episode = 0

    def next_episode(self):
        self.episode += 1
        self.recorder.append(Episode())
        self.last_obs = None

    # Train model with recorded game data
    def train(self):
        loss = np.array([0., 0., 0., 0.])
        for ep in self.recorder:
            loss += self._train(
                ep.screen_input[:ep.current_step],
                ep.action_input[:ep.current_step],
                ep.unit_input[:ep.current_step],
                get_discounted_rewards(ep.rewards[:ep.current_step], discount_rate=DISCOUNT_RATE),
                ep.nonspatial_action[:ep.current_step],
                ep.spatial_action[:ep.current_step],
                ep.screen_used[:ep.current_step]
            )
        return loss / len(self.recorder)

    def _train(self, screens_input, action_input, select_input, reward, action, screen_action, screen_used):
        _entropy = _policy_loss = _value_loss = 0.

        with tf.GradientTape() as tape:
            spatial_policy, ns_policy, value = self.model([screens_input, action_input, select_input])
            value = K.squeeze(value, axis=1)

            ns_action_one_hot = K.one_hot(action, len(ACTION_OPTIONS))
            screen_action_one_hot = K.one_hot(screen_action, SCREEN_SIZE * SCREEN_SIZE)

            value_loss = .5 * K.square(reward - value)

            entropy = -K.sum(ns_policy * K.log(ns_policy + 1e-10), axis=1) - \
                       K.sum(spatial_policy * K.log(spatial_policy + 1e-10), axis=1)
            ns_log_prob = K.log(K.sum(ns_policy * ns_action_one_hot, axis=1) + 1e-10)
            spatial_log_prob = K.log(K.sum(spatial_policy * screen_action_one_hot, axis=1) + 1e-10)
            advantage = reward - K.stop_gradient(value)

            # Mask out spatial_log_prob when the action taken did not use the screen
            policy_loss = -(ns_log_prob + spatial_log_prob * screen_used) * advantage - entropy * ENTROPY_RATE

            total_loss = policy_loss + value_loss

            _entropy = K.mean(entropy)
            _policy_loss = K.mean(K.abs(policy_loss))
            _value_loss = K.mean(value_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        global_norm = tf.linalg.global_norm(gradients)
        print(tf.linalg.global_norm(gradients))
        gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_MAX)  # Prevents exploding gradients...I think
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return [float(_value_loss), float(_policy_loss), float(_entropy), global_norm]

    def strip_reshape(self, arr):
        return np.reshape(arr, tuple(s for s in arr.shape if s > 1))

    # Call with game end step and the outcome from the environment
    def step_end(self, obs, outcome):
        last_reward = self.calc_reward(obs, self.last_obs, outcome=outcome)
        self.recorder[self.episode].reward_last_step(last_reward)

    # Takes a state and returns an action, also updates step information
    def step(self, obs, training=True):
        episode = self.recorder[self.episode]

        if self.last_obs:
            last_reward = self.calc_reward(obs, self.last_obs)
            episode.reward_last_step(last_reward)

        screens_input, action_input, select_input = self.build_inputs_from_obs(obs)
        spatial_action_policy, ns_action_policy, value = self.model([screens_input, action_input, select_input])

        # Remove dimensions with length 1
        spatial_action_policy = self.strip_reshape(spatial_action_policy)
        ns_action_policy = self.strip_reshape(ns_action_policy)

        if training:
            try:
                screen_choice = np.random.choice(SCREEN_SIZE * SCREEN_SIZE,
                                                 p=spatial_action_policy / np.sum(spatial_action_policy))
            except Exception as e:
                print('Error in %s' % self.name)
                raise
        else:
            screen_choice = np.argmax(spatial_action_policy)

        screen_x = screen_choice // SCREEN_SIZE
        screen_y = screen_choice % SCREEN_SIZE

        if training:
            # Select from probability distribution
            choice = np.random.choice(len(ns_action_policy), p=ns_action_policy)
        else:
            # Select highest probability
            choice = int(np.argmax(ns_action_policy))

        action = ACTION_OPTIONS[choice]
        build_args = []
        # Build action
        for arg in action['args']:
            if arg == 'screen':
                build_args.append([screen_x, screen_y])
            elif arg == 'screen_rect':
                build_args.append([np.max([(screen_x - SELECT_SIZE), 0]),
                                   np.max([(screen_y - SELECT_SIZE), 0])])
                build_args.append([np.min([(screen_x + SELECT_SIZE), SCREEN_SIZE-1]),
                                   np.min([(screen_y + SELECT_SIZE), SCREEN_SIZE-1])])
            elif type(arg) is int:
                build_args.append([arg])
            else:
                raise KeyError('Unrecognized function argument: %s' % arg)

        self.recorder[self.episode].save_step(
            (screens_input, action_input, select_input),
            (spatial_action_policy, ns_action_policy, value),
            choice,
            screen_choice,
            ('screen' in action['args'] or 'screen_rect' in action['args'])
        )

        self.last_obs = obs
        return actions.FunctionCall(action['id'], build_args)

    def build_inputs_from_obs(self, obs):
        screens_input = np.zeros((SCREEN_DEPTH, SCREEN_SIZE, SCREEN_SIZE), dtype=np.float32)
        # Transpose feature screens because spatial observations are (y,x) coordinates, everything else is (x,y)
        for ndx, name in enumerate(INPUT_SCREENS):
            if name == 'player_relative':
                screens_input[ndx] = self.convert_player_ids_vect(np.array(obs.observation['feature_screen'][name]))
            elif name == 'unit_type':
                unit_types = np.array(obs.observation['feature_screen'][name])
                screens_input[ndx] = self.convert_unit_ids_vect(unit_types)
            elif name == 'unit_hit_points':
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]) / UNIT_HP_SCALE
            else:
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]) / getattr(features.SCREEN_FEATURES, name).scale

        screens_input = np.reshape(screens_input, (1, SCREEN_SIZE, SCREEN_SIZE, SCREEN_DEPTH))

        # Available actions as array of 1 and 0
        action_input = np.array([
            (0. if
             act_info['id'] not in obs.observation['available_actions'] or
             (act_info['id'] == actions.FUNCTIONS.select_unit.id and act_info['args'][1] >= len(obs.observation['multi_select']))
             else 1.)
            for act_info in ACTION_OPTIONS
        ], dtype=np.float32)
        action_input = np.reshape(action_input, (1, len(ACTION_OPTIONS)))

        # Normalizes the unit select tensor and removes fields
        def convert_select_tensor(x):
            return np.array([
                self.convert_unit_ids(x[0]),
                self.convert_player_ids(x[1]),
                x[2] / UNIT_HP_SCALE
            ], dtype=np.float32)

        # Selected units
        select_input = np.zeros((MAX_UNIT_SELECT, UNIT_TENSOR_LENGTH), dtype=np.float32)
        for ndx, unit in enumerate(obs.observation['multi_select']):
            select_input[ndx] = convert_select_tensor(unit)
        select_input = np.reshape(select_input, (1, MAX_UNIT_SELECT * UNIT_TENSOR_LENGTH))

        return screens_input, action_input, select_input

    def calc_reward(self, obs, obs_prev, outcome=0.):
        rw = self.reward_weights

        score = obs.observation['score_by_category']
        score_prev = obs_prev.observation['score_by_category']
        # Difference in army killed minerals and vespene cost minus diff in lost minerals and vespene since last state
        enemy_killed_value = (score[1][1] - score_prev[1][1]) + VESPENE_SCALING*(score[2][1] - score_prev[2][1])
        friendly_killed_value = (score[3][1] - score_prev[3][1]) + VESPENE_SCALING*(score[4][1] - score_prev[4][1])
        diff_value = rw['enemy_killed_value'] * enemy_killed_value - rw['friendly_killed_value'] * friendly_killed_value

        score = obs.observation['score_by_vital']
        score_prev = obs_prev.observation['score_by_vital']
        # Difference in damage dealt minus damage taken since last state
        damage_given = score[0][0] - score_prev[0][0]
        damage_taken = score[1][0] - score_prev[1][0]

        diff_damage = rw['damage_given'] * damage_given - rw['damage_taken'] * damage_taken

        reward = .005 * rw['killed_value'] * diff_value + .01 * rw['damage'] * diff_damage + rw['outcome'] * outcome * .5
        return reward

    def build_model(self, screen_width, screen_height, screen_depth, select_input_length, action_size, training=True):
        K.set_floatx('float32')

        # Inputs
        screen_input = Input(shape=(screen_width, screen_height, screen_depth), dtype='float32')
        action_input = Input(shape=(action_size,), dtype='float32')
        select_input = Input(shape=(MAX_UNIT_SELECT * select_input_length,), dtype='float32')

        screen_part = TimeDistributed(Conv2D(screen_depth, 5, strides=1, padding='same'))(screen_input)
        screen_part = TimeDistributed(BatchNormalization())(screen_part)
        screen_part = TimeDistributed(Activation('relu'))(screen_part)
        screen_part = TimeDistributed(Conv2D(screen_depth, 3, strides=1, padding='same'))(screen_part)
        screen_part = TimeDistributed(BatchNormalization())(screen_part)
        screen_part = TimeDistributed(Activation('relu'))(screen_part)

        action_1 = TimeDistributed(Dense(screen_width*screen_height, use_bias=True, activation='relu', name='ingrid'))(action_input)
        action_1 = TimeDistributed(Reshape((screen_width, screen_height, 1)))(action_1)

        select_1 = TimeDistributed(Dense(screen_width*screen_height, use_bias=True, activation='relu', name='steve'))(select_input)
        select_1 = TimeDistributed(Reshape((screen_width, screen_height, 1)))(select_1)

        core = TimeDistributed(Concatenate(axis=3)([screen_part, action_1, select_1]))
        core = ConvLSTM2D(1, 5, strides=1, padding='same', activation='relu', training=training)(core)
        # core = Conv2D(10, 5, strides=1, padding='same')(core)
        # core = BatchNormalization()(core)
        # core = Activation('relu')(core)
        # core = Conv2D(4, 5, strides=1, padding='same')(core)
        # core = BatchNormalization()(core)
        # core = Activation('relu')(core)

        action_policy = TimeDistributed(Conv2D(1, 3, strides=2, padding='same', activation='relu'))(core)
        action_policy = TimeDistributed(Flatten())(action_policy)
        action_policy = TimeDistributed(Dense(action_size * 2, use_bias=True, activation='relu'))(action_policy)
        if training:
            action_policy = TimeDistributed(Dropout(DROPOUT_RATE))(action_policy)
        action_policy = TimeDistributed(Dense(action_size * 2, use_bias=True, activation='relu'))(action_policy)
        if training:
            action_policy = TimeDistributed(Dropout(DROPOUT_RATE))(action_policy)
        action_policy = TimeDistributed(Dense(action_size))(action_policy)
        # Mask out unavailable actions and softmax
        action_policy = K.exp(action_policy) * action_input / (K.sum(K.exp(action_policy) * action_input))

        value = TimeDistributed(Conv2D(1, 5, strides=3, activation='relu'))(core)
        value = TimeDistributed(Flatten())(value)
        if training:
            value = TimeDistributed(Dropout(DROPOUT_RATE))(value)
        value = TimeDistributed(Dense(50, use_bias=True, activation='relu'))(value)
        value = TimeDistributed(Dense(1))(value)

        # Concat in the action policy to inform the screen policy
        action_policy_dense = TimeDistributed(Dense(screen_width*screen_height, use_bias=True, activation='relu'))(K.stop_gradient(action_policy))
        action_policy_dense = TimeDistributed(Reshape((screen_width, screen_height, 1)))(action_policy_dense)
        screen_core = TimeDistributed(Concatenate(axis=3))([core, action_policy_dense])
        screen_policy = TimeDistributed(Conv2D(5, 3, padding='same'))(screen_core)
        screen_policy = TimeDistributed(BatchNormalization())(screen_policy)
        screen_policy = TimeDistributed(Activation('relu'))(screen_policy)
        screen_policy = TimeDistributed(Conv2D(1, 3, padding='same'))(screen_policy)
        screen_policy = TimeDistributed(Flatten())(screen_policy)
        screen_policy = TimeDistributed(Activation('softmax'))(screen_policy)

        model = Model([screen_input, action_input, select_input], [screen_policy, action_policy, value])

        return model
