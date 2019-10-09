
# Inputs:
#  Feature map of enemy and friendly unit health/dps/speed

# Outputs:
#  List of actions and corresponsing predicted win probabilities for that action
#

# Define a sub-layer of the network, work backwards to determing ideal number of copies of it in the hidden layer
#  Cov-net to same size

# Action is [2+maxargs x max actions] matrix of probabilities
# Another part of the out matrix is the arguments which may not get used
#
"""
probability      action      arg1   arg2   arg3   ...
probability2     action      arg1   arg2   arg3   ...
probability3     action      arg1   arg2   arg3   ...
...
"""
# Only action type is filtered through LSTM?

# How do I give feedback for individual actions?
# How do I use the game result [+1, -1]?  Do I use it against every action taken that game?
# Does that mean I have to save the input and output at every step?


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
        # actions.FUNCTIONS.select_control_group.id,
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
    MIN_X = 22
    MAX_X = 44
    HEIGHT_X = MAX_X - MIN_X

    MIN_Y = 20
    MAX_Y = 36
    HEIGHT_Y = MAX_Y - MIN_Y

    def __init__(self, model=None):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        # (input, output) tuples for the current scenario
        self.storage = []

        if not model:
            self.model = SpicyModel()
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

        # Build input
        inputs = np.empty((10, self.HEIGHT_X, self.HEIGHT_Y))

        # Add screens
        inputs[0] = np.array(obs.observation['screen'], dtype=np.float32)
        '''
        obs.observation['screen'][1]
        
        Layers:
        7 screens: player_id, player_relative, unit_type, selected, hit_points, unit_density, unit_density_aa  (22, 16)
        selected_unit or multi_select  (n,7)
        available_actions
        last_action
        '''

        output_actions, output_args = self.model.call(inputs, training=True)

        # Filter out unavailable actions
        # TODO: Do this the proper numpy way
        for ndx in range(len(output_actions)):
            if ndx not in obs.observation['available_actions']:
                output_actions[ndx] = 0.0
        action_id = self.action_options[np.argmax(output_actions)]

        action_args = []
        iter_args = iter(output_args)
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name in ('screen', 'screen2'):
                action_args.append([next(iter_args) * (self.MAX_X-self.MIN_X) + self.MIN_X,
                                    next(iter_args) * (self.MAX_Y-self.MIN_Y) + self.MIN_Y])
            elif arg.name == 'minimap':
                raise ('Unused function argument type: %s' % arg.name)
            elif arg.name == 'queued':
                action_args.append([0])
            else:
                raise('Unknown function argument type: %s' % arg.name)

        # TODO: Add LSTM for the action
        
        self.storage.append((inputs, output_actions, output_args))
        return actions.FunctionCall(action_id, action_args)

    def build_model(self, reuse, dev, ntype):
        with tf.variable_scope(self.name) and tf.device(dev):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks
            self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

            # Build networks
            net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
            self.spatial_action, self.non_spatial_action, self.value = net

            # Set targets and masks
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                           name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                              name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783
            # Policy loss and value loss
            action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.value_target - self.value)
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = - tf.reduce_mean(self.value * advantage)
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

            # TODO: policy penalty
            loss = policy_loss + value_loss

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=100)

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
