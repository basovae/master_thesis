from .ddpg import DDPG, DDPGTrainer
#from .dqn import DeepQLearning, DeepQLearningTrainer
from .networks import NeuralNetwork

__all__ = [
    'DDPG',
    'DDPGTrainer',
    #'DeepQLearning',
    #'DeepQLearningTrainer',
    'NeuralNetwork',
]
