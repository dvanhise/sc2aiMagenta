from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from itertools import count, combinations
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from spicy_agent import SpicyAgent
from maps import MapCM, MapCMI


FLAGS = flags.FLAGS

point_flag.DEFINE_point("feature_screen_size", "64",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", False,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", True, "Whether to disable Fog of War.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "CodeMagentaIsland", "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")


AGENT_COUNT = 3


def run(players, map_name, visualize):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(
            map_name=map_name,
            battle_net_map=FLAGS.battle_net_map,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                action_space=FLAGS.action_space,
                use_feature_units=FLAGS.use_feature_units,
                use_raw_units=FLAGS.use_raw_units),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        # Initialize agents
        agents = [SpicyAgent() for _ in range(AGENT_COUNT)]

        for _ in count(1):
            # Full round robin
            for iteration, (player1, player2) in enumerate(combinations(agents, 2)):
                player1.reset()
                player2.reset()

                # 16 steps = 1 second of game time, running 1 frame every 8 steps, is 2 frames per second
                # End right before scenario timer or it sometimes crashes
                run_scenario_loop([player1, player2], env, max_frames=2*119)


def run_scenario_loop(agents, env, max_frames=0):
    total_frames = 0
    start_time = time.time()

    config = tf.ConfigProto(allow_soft_placement=True)
    sessions = (tf.Session(config=config) for _ in agents)

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, sess, obs_spec, act_spec in zip(agents, sessions, observation_spec, action_spec):
        agent.setup(sess, obs_spec, act_spec)

    done = False
    rewards_total = (0., 0.)
    states = env.reset()
    Q = np.zeros([1100101010, 10])  # states x actions
    while True:
        total_frames += 1
        actions = [agent.step(state) for agent, state in zip(agents, states)]

        prev_states = states
        states = env.step(actions)
        rewards = [agent.calc_reward(state, prev_state) for agent, state, prev_state in zip(agents, states, prev_states)]
        done = states[0].last() or (max_frames and total_frames >= max_frames)

        # Update Q table

        Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})

        rewards_total = [rt + r for rt, r in zip(rewards_total, rewards)]

        if done:
            break

    for sess in sessions:
        sess.close()
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds at %.3f fps" % (elapsed_time, total_frames / elapsed_time))


def main(unused_argv):
    """Run an agent."""
    if FLAGS.trace:
        stopwatch.sw.trace()
    elif FLAGS.profile:
        stopwatch.sw.enable()

    players = [
        sc2_env.Agent(sc2_env.Race['terran'], 'player1'),
        sc2_env.Agent(sc2_env.Race['terran'], 'player2')
    ]

    run(players, FLAGS.map, visualize=True)

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    # Register maps
    MapCM()
    MapCMI()
    globals()['CodeMagenta'] = type('CodeMagenta', (MapCM,), dict(filename='CodeMagenta'))
    globals()['CodeMagentaIsland'] = type('CodeMagentaIsland', (MapCMI,), dict(filename='CodeMagentaIsland'))

    app.run(main)
