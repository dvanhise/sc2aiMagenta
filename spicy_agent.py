from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.lib import units
from pysc2.lib import features
import numpy as np
import tensorflow as tf
from spicy_config import *
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
        None,
        units.Terran.Marine,
        units.Terran.Marauder,
        units.Terran.Hellion
    ]

    # Available argument types and their offset in the arg tensor
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
    MAX_UNIT_SELECT = 8

    LEARNING_RATE = .2
    DISCOUNT_RATE = .9
    EXPLORATION_RATE = .3
    VESPENE_SCALING = 1.5
    UNIT_HP_SCALE = 200  # 1600 by default

    def __init__(self, model=None):
        self.reward = 0
        self.episodes = 0
        self.steps = -1
        self.obs_spec = None
        self.action_spec = None
        self.sess = None

        self.recorder = []

        if not model:
            self.model = SpicyModel(self.SCREEN_SIZE, self.SCREEN_SIZE, self.SCREEN_DEPTH,
                                    len(self.action_options) + self.MAX_UNIT_SELECT * self.SELECT_SIZE,
                                    len(self.action_options) + len(self.arg_options))
        else:
            self.model = model

        # How to convert blizzard unit and building IDs to our subset of units
        def convert_unit_ids(x):
            if x in self.unit_options:
                return self.unit_options.index(x) / len(self.unit_options)
            return self.unit_options.index(None) / len(self.unit_options)
        self.convert_unit_ids = convert_unit_ids
        self.convert_unit_ids_vect = np.vectorize(convert_unit_ids)

        # How to convert 'player_relative' data
        def convert_player_ids(x):
            if x == 3:
                return 0.
            elif x == 5:
                return 1.
            else:
                raise ValueError('That should not happen')
        self.convert_player_ids = convert_player_ids
        self.convert_player_ids_vect = np.vectorize(convert_player_ids)

        self.graph = tf.Graph()
        with self.graph.as_default():

            value_target = tf.placeholder("float", [None])
            global_step = tf.placeholder("int", [None])
            screen_input = tf.placeholder("float", [self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE])
            ns_input = tf.placeholder("float", [len(self.action_options) + self.MAX_UNIT_SELECT * self.SELECT_SIZE])

            # spatial_action_policy, ns_action_policy, value = \
            #     self.build_model(screen_input, ns_input, len(self.action_options) + len(self.arg_options))

            spatial_action_policy, ns_action_policy, value = self.model.call(screen_input, ns_input)

            action_probs, action_args = tf.split(ns_action_policy, [len(self.action_options), len(self.arg_options)])

            optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY_RATE)

            entropy = tf.reduce_sum(ns_action_policy * tf.log(ns_action_policy) *
                                    spatial_action_policy * tf.log(spatial_action_policy))
            advantage = value_target - value

            # Is policy loss correct here?  Why is there a negative?
            p_loss = -tf.reduce_sum(tf.log(action_probs) * tf.log(spatial_action_policy) * advantage)
            v_loss = tf.reduce_mean(tf.square(advantage))
            total_loss = p_loss + v_loss - entropy*ENTROPY_RATE

            optimizer.minimize(total_loss, global_step=global_step)

    def setup(self, sess, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.sess = sess

    def reset(self):
        self.recorder = []
        self.steps = -1
        self.episodes += 1

    def train(self):
        total_value_loss = 0
        total_policy_loss = 0

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()

            value_target = 0
            for state in reversed(self.recorder):
                value_target = state.reward + DISCOUNT_RATE * value_target

                feed_dict = {
                    'value_target': value_target,
                    'global_step': self.episodes,
                    'screen_input': state.inputs['spatial'],
                    'ns_input': state.inputs['nonspatial']
                }
                _, p_loss, v_loss, entropy = session.run(feed_dict=feed_dict)
                total_policy_loss += p_loss
                total_value_loss += v_loss

        print('Step %d: Average loss - policy: %.4f  value: %.4f' %
              (self.episodes, total_policy_loss/len(self.recorder), total_value_loss/len(self.recorder)))

    # Takes a state and returns an action, also updates step information
    def step(self, obs):
        self.steps += 1
        # On first step, center camera in map (maybe best in map editor?)
        # if self.steps == 1:
        #     return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        #     return actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [32, 32])

        # Calculate reward of previous step and update it's state
        if self.steps != 0:
            reward = self.calc_reward(obs, self.recorder[self.steps-1].obs)
            self.recorder[self.steps-1].update(obs, self.calc_reward(obs, reward))

        screens_input, ns_input = self.build_inputs_from_obs(obs)

        print('Input Shapes: %s -- %s' % (screens_input.shape, ns_input.shape))
        spatial_action_policy, ns_action_policy, value = self.build_model(screens_input, ns_input, len(self.action_options) + len(self.arg_options))
        print('Output Shapes: %s -- %s' % (spatial_action_policy.shape, ns_action_policy.shape))

        # Clip the values to 1
        #   I don't think I should need to do this with sigmoid activation functions, yet I get errors
        action_probs = np.clip(ns_action_policy, None, .9999999)[:len(self.action_options)]
        action_args = np.clip(ns_action_policy, None, .9999999)[len(self.action_options):]

        screen_max = np.argmax(spatial_action_policy)
        screen_x = screen_max // spatial_action_policy.shape[1]
        screen_y = screen_max % spatial_action_policy.shape[1]

        # Filter out unavailable actions before choosing the best one
        for ndx in range(len(action_probs)):
            if self.action_options[ndx] not in obs.observation['available_actions']:
                action_probs[ndx] = 0.

        # Compute softmax with unavailable actions removed
        action_probs = tf.math.softmax(action_probs)

        # Use argmax for playing, use choice with probabilities for training
        # action_id = self.action_options[int(np.argmax(action_probs))]
        action_id = self.action_options[np.random.choice(range(action_probs), p=action_probs)]

        build_args = []
        # Build action
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name == 'screen1':
                build_args.append([screen_x + 3, screen_y])
            # screen2 is only part of rect_select so use preset size based on screen1 to avoid the variable
            elif arg.name == 'screen2':
                build_args.append([(screen_x + self.SELECT_SIZE) % self.SCREEN_SIZE,
                                   (screen_y + self.SELECT_SIZE) % self.SCREEN_SIZE])
            elif arg.name == 'select_unit_id':
                select_unit_id = action_args[self.arg_options.index('select_unit_id')]
                local_index = math.floor(select_unit_id * len(self.unit_options))
                build_args.append([self.unit_options[local_index]])
            elif arg.name == 'select_add':
                select_add = action_args[self.arg_options.index('select_add')]
                build_args.append([math.floor(select_add * 2)])
            elif arg.name in 'select_unit_act':
                select_unit_act = action_args[self.arg_options.index('select_unit_act')]
                build_args.append([math.floor(select_unit_act * 4)])
            elif arg.name == 'select_point_act':
                select_point_act = action_args[self.arg_options.index('select_point_act')]
                build_args.append([math.floor(select_point_act * 4)])
            # Always set queued arg as false
            elif arg.name == 'queued':
                build_args.append([0])
            else:
                raise('Unrecognized function argument type: %s' % arg.name)

        # TODO: Add LSTM

        self.recorder.append(State(obs, (screens_input, ns_input), (spatial_action_policy, ns_action_policy, value)))
        print("Action: %d, Args: %s" % (action_id, action_args))
        return actions.FunctionCall(action_id, action_args)

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
                screens_input[ndx] = np.array(obs.observation['feature_screen'][name]) / features.SCREEN_FEATURES[name].scale

        screens_input = screens_input.T
        screens_input = np.reshape(screens_input, (1, self.SCREEN_DEPTH, self.SCREEN_SIZE, self.SCREEN_SIZE))

        # Normalizes the unit select tensor and removes fields
        def convert_select_tensor(x):
            return np.array([
                self.convert_unit_ids(x[0]),
                self.convert_player_ids(x[1]),
                x[2] / self.UNIT_HP_SCALE
            ], dtype=np.float32)

        # Available actions
        act_input = np.zeros(len(self.action_options))
        available_actions = obs.observation['available_actions']
        for ndx, act_id in enumerate(self.action_options):
            act_input[ndx] = (1. if act_id in available_actions else 0.)

        # Selected units
        multi_select = np.array(convert_select_tensor(obs.observation['multi_select']), dtype=np.float32)
        ms = np.resize(multi_select, (self.UNIT_TENSOR_LENGTH, min(self.MAX_UNIT_SELECT, multi_select.shape[1])))

        ns_input = np.concatenate((act_input, ms))

        return screens_input, ns_input

    def calc_reward(self, obs, obs_prev):
        if obs.last():
            return 0.

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

    @tf.function
    def build_model(self, screen, ns_input, action_size):

        sconv1 = tf.layers.conv2d(screen,
                                  num_outputs=16,
                                  kernel_size=5,
                                  stride=1,
                                  scope='sconv1')
        sconv2 = tf.layers.conv2d(sconv1,
                                  num_outputs=32,
                                  kernel_size=3,
                                  stride=1,
                                  scope='sconv2')
        ns_input_fc = tf.layers.fully_connected(ns_input,
                                                num_outputs=256,
                                                activation_fn=tf.relu,
                                                scope='info_fc')

        # Compute spatial actions
        state = tf.concat([sconv2, ns_input_fc], axis=3)
        spatial_action_policy = tf.layers.conv2d(state,
                                                 num_outputs=1,
                                                 kernel_size=1,
                                                 stride=1,
                                                 activation_fn=None,
                                                 scope='spatial_action')
        spatial_action_policy = tf.nn.softmax(tf.layers.flatten(spatial_action_policy))

        # Compute non spatial actions and value
        state_fc = tf.layers.fully_connected(state,
                                             num_outputs=256,
                                             activation_fn=tf.nn.relu,
                                             scope='feat_fc')
        ns_action_policy = tf.layers.fully_connected(state_fc,
                                                     num_outputs=action_size,
                                                     activation_fn=None,
                                                     scope='non_spatial_action')
        value = tf.layers.fully_connected(state_fc,
                                          num_outputs=1,
                                          activation_fn=None,
                                          scope='value')

        return spatial_action_policy, ns_action_policy, value


class State:
    def __init__(self, observation, inputs, outputs):
        self.obs = observation
        self.inputs = {
            'spatial': inputs[0],
            'nonspatial': inputs[1]
        }
        self.outputs = {
            'spatial': outputs[0],
            'nonspatial': outputs[1],
            'value': outputs[2]
        }
        self.next_obs = None
        self.reward = None
        self.done = True

    def update(self, next_observation, reward):
        self.next_obs = next_observation
        self.reward = reward
        self.done = False


