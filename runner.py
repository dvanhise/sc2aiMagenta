import time
from itertools import count, combinations

from absl import app
from absl import flags

from pysc2.env import available_actions_printer, sc2_env
from pysc2.lib import point_flag, stopwatch

from spicy_agent import SpicyAgent
from maps import MapCM, MapCMI
import warnings

import os


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


AGENT_COUNT = 4


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
            ensure_available_actions=True,
            step_mul=FLAGS.step_mul,
            realtime=True,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

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

        for gen in count(1):
            print('>>> Begin generation %d' % gen)

            # Full round robin
            for iteration, (player1, player2) in enumerate(combinations(agents, 2)):
                player1.reset()
                player2.reset()

                # 16 steps = 1 second of game time, running 1 frame every 8 steps is 2 frames per second
                # End right before scenario timer or it sometimes crashes
                print('>>> Start game loop between %s and %s' % (player1.name, player2.name))
                run_scenario_loop([player1, player2], env)

                print('>>> Train agents')
                player1.train()
                player2.train()

            for agent in agents:
                agent.model.save_weights('./save/%s.tf' % agent.name)


def run_scenario_loop(agents, env, max_frames=0):
    total_frames = 0
    episode = 1
    start_time = time.time()

    states = env.reset()
    while True:
        total_frames += 1
        actions = [agent.step(state) for agent, state in zip(agents, states)]
        states = env.step(actions)

        if states[0].last() or states[1].last() or (max_frames and total_frames >= max_frames):
            print('Detected end of game')
            if episode >= 1:
                break
            episode += 1
            total_frames = 0
            states = env.reset()

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
    # Good programming practice: Silence all warnings :P
    warnings.filterwarnings("ignore")

    # Register maps
    MapCM()
    MapCMI()
    globals()['CodeMagenta'] = type('CodeMagenta', (MapCM,), dict(filename='CodeMagenta'))
    globals()['CodeMagentaIsland'] = type('CodeMagentaIsland', (MapCMI,), dict(filename='CodeMagentaIsland'))

    app.run(main)
