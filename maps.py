from pysc2.maps import lib


class MapCM(lib.Map):
    directory = "."
    filename = "CodeMagenta"
    players = 2
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


class MapCMI(lib.Map):
    directory = "."
    filename = "CodeMagentaIsland"
    players = 2
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8

