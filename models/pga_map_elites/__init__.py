from .pga_map_elites import main , config
#from .official_logic import PGAMAPElites, TD3Trainer
from .networks import Critic, Actor
from .archive import CVTArchive
from .replay_buffer import ReplayBuffer
from .variational_operators import variation

__all__ = [
    'main',
    'config',
    'Critic',
    'Actor',
    'CVTArchive',
    'ReplayBuffer',
    'variation',
]
