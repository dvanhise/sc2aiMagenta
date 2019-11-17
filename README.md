## MagentaAI
Trains AI agents using reinforcement learning to play a micro focused Starcraft 2 minigame in the pysc2 environment.

### Requirements:
* Install Starcraft 2: Legacy of the Void and set environment variable *SC2PATH* to the install directory
* Python 3
* Install packages in requirements.txt

### Usage
Train agents: `python runner.py --mode=train --realtime=False`

Demonstrate agents: `python runner.py --mode=test --vis=True`

Play against trained agent: `python runner.py --mode=play`