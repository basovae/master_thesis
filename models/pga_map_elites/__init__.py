from .pga_map_elites import main , config
#from .official_logic import PGAMAPElites, TD3Trainer
from .networks import Critic, Actor
from .replay_buffer import ReplayBuffer

__all__ = [
    'main',
    'config',
    'Critic',
    'Actor',
    'ReplayBuffer',
]
