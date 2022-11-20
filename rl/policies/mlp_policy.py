import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rl.networks.mlp import Mlp
from util.etc import topk_filter, get_device

class MlpPolicy(nn.Module):
    def __init__(self, embedding_size, hidden_layer_sizes, trainset_size, replace=False):
        super().__init__()
        self.mlp = Mlp(embedding_size, hidden_layer_sizes, trainset_size)
        self.replace = replace

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
        embeddings, action_mask = states
        logits = self.forward(embeddings)
        logits = torch.where(action_mask.bool(), logits, torch.full_like(logits, float('-inf'), device=get_device()))

        if max_action:
            indices = torch.argmax(logits, dim=-1)
        else:
            indices = Categorical(logits=logits).sample()
        return (indices, self.replace), logits

    def get_weight_sum(self):
        return self.mlp.get_weight_sum()