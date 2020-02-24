from pysc2.lib import actions
from pysc2.lib import units


# Running parameters
AGENT_COUNT = 5             # Number of agents to put in the training group
EPISODES_PER_MATCH = 10     # Number of consecutive games to play in each training matchup
MAX_TIMESTEPS = 90          # Max timesteps in a game

# Hyperparameters
LEARNING_RATE = .0001
ENTROPY_RATE = .002
DISCOUNT_RATE = .92         # Timestep reward discount
GRADIENT_CLIP_MAX = 200.
DROPOUT_RATE = .5

# Various details
MAX_UNIT_SELECT = 5         # The highest index unit in a selection that can be selected individually
CONTROL_GROUPS_AVAIL = 3    # The number of control groups the agent can utilize
VESPENE_SCALING = 1.25      # Ratio relative to minerals to scale unit values with determining rewards
UNIT_HP_SCALE = 200         # Max HP value to scale input unit HP
SELECT_SIZE = 4             # Radius in ??? units around screen point to do a "select in rectangle"
UNIT_TENSOR_LENGTH = 3

# Tensor dimention things
INPUT_SCREENS = ['player_relative', 'unit_type', 'selected', 'unit_hit_points',
                 'unit_hit_points_ratio', 'active', 'unit_density', 'unit_density_aa']
SCREEN_DEPTH = len(INPUT_SCREENS)
SCREEN_SIZE = 64            # "Pixels" of detail in x and y axis off screen data

ACTION_OPTIONS = [
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
    # *({'id': actions.FUNCTIONS.select_control_group.id, 'args': [act, id]} for id in range(CONTROL_GROUPS_AVAIL) for act in range(3)),
    # select_unit_act options: Select, deselect, select all of type, deselect all of type
    *({'id': actions.FUNCTIONS.select_unit.id, 'args': [act, id]} for id in range(MAX_UNIT_SELECT) for act in range(2)),
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

# Master list of units agent "knows" about
UNIT_OPTIONS = [
    units.Terran.Marine,
    units.Terran.Marauder,
    units.Terran.Hellion
]

