import time
from itertools import count, combinations
import os
import logging
from datetime import datetime
import random

from absl import app
from absl import flags

from pysc2.env import available_actions_printer, sc2_env
from pysc2.lib import point_flag, stopwatch

from spicy_agent import SpicyAgent
from maps import MapCM, MapCMI


FLAGS = flags.FLAGS

# train = AIs play each other, they explore new actions and train models
# test = AIs play each other, they choose best possible actions
# play = AI agent plays against human
flags.DEFINE_enum('mode', 'train', ['train', 'test', 'play'], 'Whether to run in training mode.')
flags.DEFINE_bool("vis", False, "Whether to show pygame feature maps window.")
flags.DEFINE_bool("realtime", True, "Whether to run in realtime as opposed to max speed.")
flags.DEFINE_string("map", "CodeMagentaIsland", "Name of a map to use.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay of each game played.")

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

flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")

AGENT_COUNT = 5


def run(players, agents, map_name):
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
            ensure_available_actions=True,
            step_mul=FLAGS.step_mul,
            realtime=FLAGS.realtime,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=FLAGS.vis) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        if FLAGS.mode == 'play':
            agent = random.choice(agents)
            run_game_loop([agent], env)
            if FLAGS.save_replay:
                env.save_replay('./replays/%s_%s-%s_%s.SC2Replay' %
                                (FLAGS.map, agent.name, 'human', datetime.now().strftime('%Y%m%d%H%M%S')))
            return

        for gen in count(1):
            print('>>> Begin generation %d' % gen)

            # Full round robin
            total_time = 0.
            total_entropy = 0.
            total_value_loss = 0.
            total_policy_loss = 0.
            games_played = 0
            for iteration, (player1, player2) in enumerate(combinations(agents, 2)):
                player1.reset()
                player2.reset()

                print('>>> Start game loop between %s and %s' % (player1.name, player2.name))
                t_start = time.time()
                total_time += run_game_loop([player1, player2], env)
                print('Game completed in %.2fs' % (time.time() - t_start))
                games_played += 1

                if FLAGS.mode == 'train':
                    print('>>> Train agents')
                    t_start = time.time()
                    value_loss, policy_loss, entropy = player1.train()
                    value_loss2, policy_loss2, entropy2 = player2.train()
                    print('Training completed in %.2fs' % (time.time() - t_start))

                    total_value_loss += value_loss + value_loss2
                    total_policy_loss += policy_loss + policy_loss2
                    total_entropy += entropy + entropy2

                if FLAGS.save_replay:
                    env.save_replay('%s_%s-%s_%s.SC2Replay' %
                                    (FLAGS.map, player1.name, player2.name, datetime.now().strftime('%Y%m%d%H%M%S')))

            logging.info('Generation %d complete' % gen)
            logging.info('Average game time: %.2fs' % (total_time / games_played))
            logging.info('Average value loss: %.3f' % (total_value_loss / (2*games_played)))
            logging.info('Average policy loss: %.3f' % (total_policy_loss / (2*games_played)))
            logging.info('Average entropy: %.3f' % (total_entropy / (2*games_played)))

            if FLAGS.mode == 'train':
                for agent in agents:
                    agent.model.save_weights('./save/%s.tf' % agent.name)


def run_game_loop(agents, env, max_frames=0):
    total_frames = 0
    episode = 1
    start_time = time.time()

    states = env.reset()
    while True:
        total_frames += 1
        actions = [agent.step(state, training=(FLAGS.mode == 'train')) for agent, state in zip(agents, states)]
        states = env.step(actions)

        if states[0].last() or states[1].last() or (max_frames and total_frames >= max_frames):
            if episode >= 1:
                break
            episode += 1
            total_frames = 0
            states = env.reset()

    elapsed_time = time.time() - start_time
    print("Took %.3f seconds at %.3f fps" % (elapsed_time, total_frames / elapsed_time))
    return elapsed_time


def main(unused_argv):
    # Initialize agents
    agents = []
    for i in range(AGENT_COUNT):
        agent = SpicyAgent('agent%d' % i)
        path = './save/%s.tf' % agent.name
        if os.path.exists(path + '.index'):
            print('Loading from file %s' % path)
            agent.model.load_weights(path)
        else:
            print('Could not find file, randomly initilizaing model')
        agents.append(agent)

    players = [
        sc2_env.Agent(sc2_env.Race['terran'], 'player1'),
        sc2_env.Agent(sc2_env.Race['terran'], 'player2')
    ]
    run(players, agents, FLAGS.map)


if __name__ == "__main__":
    absl_logger = logging.getLogger('absl')
    absl_logger.setLevel(logging.ERROR)

    logging.basicConfig(level=logging.INFO, filename='train.log')

    # Register maps
    MapCM()
    MapCMI()
    globals()['CodeMagenta'] = type('CodeMagenta', (MapCM,), dict(filename='CodeMagenta'))
    globals()['CodeMagentaIsland'] = type('CodeMagentaIsland', (MapCMI,), dict(filename='CodeMagentaIsland'))

    app.run(main)
