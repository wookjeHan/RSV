import torch.nn as nn
from torch.nn import functional as F

from util.etc import fanin_init

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=lambda x: x,
            hidden_init=fanin_init,
            b_init_value=0.,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        return self.output_activation(preactivation)

    def get_weight_sum(self):
        weight_sum = 0
        for fc in self.fcs:
            weight_sum += fc.weight.norm(p=1)
            weight_sum += fc.bias.norm(p=1)

        weight_sum += self.last_fc.weight.norm(p=1)
        weight_sum += self.last_fc.bias.norm(p=1)

        return weight_sum.item()
