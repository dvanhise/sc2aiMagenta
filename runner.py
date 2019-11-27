import time
from itertools import count, combinations
import os
import logging
from datetime import datetime
import random
import numpy as np

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

from pysc2.env import available_actions_printer, sc2_env
from pysc2.lib import actions
import tensorflow as tf

from spicy_agent import SpicyAgent
from spicy_config import *
from maps import MapCMI


FLAGS = flags.FLAGS

# train = AIs play each other, they explore new actions and train models
# test = AIs play each other, they choose best possible actions
# play = AI agent plays against human
flags.DEFINE_enum('mode', 'train', ['train', 'test', 'play'], 'Whether to run in training mode.')
flags.DEFINE_bool("vis", False, "Whether to show pygame feature maps window.")
flags.DEFINE_bool("realtime", True, "Whether to run in realtime as opposed to max speed.")
flags.DEFINE_string("map", "CodeMagentaIsland", "Name of a map to use.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay of each game played.")

flags.DEFINE_integer("feature_screen_size", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("feature_minimap_size", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("rgb_screen_size", None,
                     "Resolution for rendered screen.")
flags.DEFINE_integer("rgb_minimap_size", None,
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


def run(players, agents):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
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
            visualize=FLAGS.vis) as main_env:
        env = available_actions_printer.AvailableActionsPrinter(main_env)

        if FLAGS.mode == 'play':
            agent = random.choice(agents)
            run_game_loop([agent], env, main_env)
            if FLAGS.save_replay:
                env.save_replay('./replays/%s_%s-%s_%s.SC2Replay' %
                                (FLAGS.map, agent.name, 'human', datetime.now().strftime('%Y%m%d%H%M%S')))
            return

        for gen in count(1):
            print('>>> Begin generation %d' % gen)

            # Full round robin
            total_time = 0.
            total_losses = [0., 0., 0.]
            games_played = 0
            wins = {agent.name: 0 for agent in agents}
            for iteration, (player1, player2) in enumerate(combinations(agents, 2)):
                player1.reset()
                player2.reset()

                print('>>> Start game loop between %s and %s' % (player1.name, player2.name))
                t_start = time.time()
                elapsed_time, results = run_game_loop([player1, player2], env, main_env)
                wins[player1.name] += results[0]
                wins[player2.name] += results[1]
                total_time += elapsed_time

                print('Game completed in %.2fs' % (time.time() - t_start))
                games_played += 1

                if FLAGS.mode == 'train':
                    print('>>> Train agents')
                    t_start = time.time()
                    total_losses += player1.train()
                    total_losses += player2.train()
                    print('Training completed in %.2fs' % (time.time() - t_start))

                for player in [player1, player2]:
                    make_reward_plot(player, save=True)
                    make_screen_plot(player, save=True)
                    make_action_plot(player, save=True)
                    # make_screen_gif(player, save=True)

                if FLAGS.save_replay:
                    env.save_replay('%s_%s-%s_%s.SC2Replay' %
                                    (FLAGS.map, player1.name, player2.name, datetime.now().strftime('%Y%m%d%H%M%S')))

            if FLAGS.mode == 'train':
                logging.info('Generation %d complete' % gen)
                logging.info('Average game time: %.2fs' % (total_time / (EPISODES_PER_MATCH * games_played)))
                logging.info('Average value loss: %.3f' % (total_losses[0] / (2 * games_played)))
                logging.info('Average policy loss: %.3f' % (total_losses[1] / (2 * games_played)))
                logging.info('Average entropy: %.3f' % (total_losses[2] / (2 * games_played)))
                for name, score in wins.items():
                    logging.info('%s: %d' % (name, score))

                for agent in agents:
                    agent.model.save_weights('./save/%s.tf' % agent.name)

            # Only run for a few generations at a time, after that performance degrades
            if gen == 2:
                break


def run_game_loop(agents, env, main_env):
    total_frames = 0
    start_time = time.time()
    cumulative_results = np.array([0, 0])

    states = env.reset()
    for episode in range(EPISODES_PER_MATCH):
        for a in agents:
            a.start_episode()
        while True:
            total_frames += 1
            acts = [agent.step(state, training=(FLAGS.mode == 'train')) for agent, state in zip(agents, states)]
            states = env.step(acts)

            results = check_for_end(states)
            if results:
                cumulative_results += np.array(results)
                for agent, state, result in zip(agents, states, results):
                    agent.step_end(state, outcome=result)
                break

        # Send reset command and run some no ops to advance the reset
        for _ in range(5):
            env.step([actions.FunctionCall(0, []), actions.FunctionCall(0, [])])
        main_env.send_chat_messages(['reset'])
        states = env.step([actions.FunctionCall(0, []), actions.FunctionCall(0, [])])

    elapsed_time = time.time() - start_time
    print("Took %.3f seconds at %.3f fps" % (elapsed_time, total_frames / elapsed_time))
    return elapsed_time, cumulative_results


def check_for_end(states):
    if states[0].last() or states[1].last():
        raise ValueError('Last frame unexpected')

    # Check for one side having no unites
    if any(state.observation['player'][8] == 0 for state in states):
        results = ((-1 if states[0].observation['player'][8] == 0 else 1),
                   (-1 if states[1].observation['player'][8] == 0 else 1))
        # Assume -1, -1 means a timeout end and give extra penalty
        if sum(results) == -2:
            return -2, -2
        return results

    return None


def make_action_plot(agent, save=False):
    action_probs = np.sum([state.outputs[1] for state in agent.recorder[0]], axis=0)
    action_probs /= len(agent.recorder[0])

    plt.clf()
    plt.plot(range(len(action_probs)), action_probs, 'bo', markersize=2)
    plt.ylabel('Mean Probability')
    plt.xlabel('Action Index')
    plt.title('%s Action Probabilities' % agent.name)
    plt.yscale('log')
    if save:
        plt.savefig('figures/action_fig_%s.png' % agent.name)
    else:
        plt.show()


def make_screen_plot(agent, save=False):
    screen_probs = np.mean([state.outputs[0] for state in agent.recorder[0]], axis=0)
    # screen_probs = agent.recorder[0][0].outputs[0]
    screen_probs = np.reshape(screen_probs, (FLAGS.feature_screen_size, FLAGS.feature_screen_size))

    plt.clf()
    plt.imshow(screen_probs.T, cmap='viridis')  # Transpose because SC2 switches x and y
    plt.colorbar()
    plt.title('%s Screen Location Probabilities' % agent.name)
    if save:
        plt.savefig('figures/screen_fig_%s.png' % agent.name)
    else:
        plt.show()


def make_reward_plot(agent, save=False):
    episode = agent.recorder[0]
    values = [float(state.outputs[2]) for state in episode]
    rewards = [state.reward for state in episode]
    discounted_rewards = agent.get_discounted_rewards(rewards)

    plt.clf()
    plt.plot(range(len(episode)), values, 'bo', markersize=2)
    plt.plot(range(len(episode)), rewards, 'ro', markersize=2)
    plt.plot(range(len(episode)), discounted_rewards, 'go', markersize=2)
    plt.ylabel('Value/Reward')
    plt.xlabel('Step')
    plt.title('%s Predicted Value and Reward' % agent.name)
    plt.legend(['Value', 'Reward', 'Discounted Reward'])
    if save:
        plt.savefig('figures/reward_fig_%s.png' % agent.name)
    else:
        plt.show()


def make_screen_gif(agent, save=False):
    if len(agent.recorder) == 0:
        return
    # Makes of gif of the screen policy probabilities for the first game of the last set
    game = np.array([np.reshape(frame.outputs[0], (FLAGS.feature_screen_size, FLAGS.feature_screen_size)).T
                     for frame in agent.recorder[0]])

    vmin = np.amin(game)
    vmax = np.amax(game)
    fig = plt.figure()
    plt.title('%s Screen Policy Probabilities' % agent.name)
    plt.axis('off')
    data = game[0]
    sns.heatmap(data, vmin=vmin, vmax=vmax, square=True, cmap='viridis')

    def init():
        plt.clf()
        sns.heatmap(np.zeros((FLAGS.feature_screen_size, FLAGS.feature_screen_size)), vmin=vmin, vmax=vmax, square=True, cmap='viridis')

    def animate(i):
        plt.clf()
        sns.heatmap(game[i], vmin=vmin, vmax=vmax, square=True, cmap='viridis')

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(game), repeat=False)
    if save:
        anim.save('figures/screen_probs_%s.gif' % agent.name, writer='imagemagick')
    else:
        plt.show()


def main(unused_argv):
    # SpicyAgent().model.summary()
    # return

    # with open('agents.yaml', 'r') as f:
    #     agents_data = yaml.safe_load(f)
    #
    # # Initialize agents
    # agents = []
    # for data in agents_data['agents']:
    #     agent = SpicyAgent(data['name'], data['weights'])
    #     path = './save/%s.tf' % agent.name
    #     if os.path.exists(path + '.index'):
    #         print('Loading from file %s' % path)
    #         agent.model.load_weights(path)
    #     else:
    #         print('Could not find file, randomly initilizaing model')
    #     agents.append(agent)

    # Initialize agents
    agents = []
    for i in range(AGENT_COUNT):
        agent = SpicyAgent('agent%d' % i)
        path = './save/%s.tf' % agent.name
        if os.path.exists(path + '.index'):
            print('Loading model weights from file %s' % path)
            agent.model.load_weights(path)
        else:
            print('Could not find saved weights, randomly initilizaing model')
        agents.append(agent)

    players = [
        sc2_env.Agent(sc2_env.Race['terran'], 'player1'),
        sc2_env.Agent(sc2_env.Race['terran'], 'player2')
    ]
    run(players, agents)


if __name__ == "__main__":
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    logging.basicConfig(level=logging.INFO, filename='train.log')

    tf.enable_eager_execution()

    # Register maps
    MapCMI()
    globals()['CodeMagentaIsland'] = type('CodeMagentaIsland', (MapCMI,), dict(filename='CodeMagentaIsland'))

    app.run(main)
