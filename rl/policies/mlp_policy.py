import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rl.networks.mlp import Mlp

class MlpPolicy(nn.Module):
    def __init__(self, trainset_size):
        super().__init__()
        self.trainset_size = trainset_size
        self.mlp = Mlp(1024, [2048], trainset_size)

    def forward(self, states):
        return self.mlp(states)

    def get_actions(self, states, max_action=False):
        '''
        returns actions
        '''
        return self.get_actions_logits(states, max_action)[0]

    def get_logits(self, states):
        '''
        returns logits
        '''
        return self.forward(states)

    def get_actions_logits(self, states, max_action=False):
        '''
        returns a tuple of actions and logits
        '''
        logits = self.get_logits(states)
        if max_action:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions, logits