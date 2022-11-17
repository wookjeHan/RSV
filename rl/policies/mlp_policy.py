import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rl.networks.mlp import Mlp
from util.etc import topk_filter

class MlpPolicy(nn.Module):
    def __init__(self, embedding_size, hidden_layer_sizes, trainset_size):
        super().__init__()
        self.mlp = Mlp(embedding_size, hidden_layer_sizes, trainset_size)

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
        return self.get_actions_logits(states)[1]

    def get_actions_logits(self, states, max_action=False):
        '''
        returns a tuple of actions and logits
        '''
        logits = self.forward(states)
        if max_action:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions, logits

    def get_weight_sum(self):
        return self.mlp.get_weight_sum()