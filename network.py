import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.action_value import DiscreteActionValue

from value_buffer import ValueBuffer

class QNetwork(nn.Module):

    def __init__(self, n_actions, n_output_dim=512):
        super().__init__()
        self.embedding_size = n_output_dim
        self.n_actions = n_actions

        self.cnn = pnn.LargeAtariCNN(n_output_channels=self.embedding_size)
        self.output = nn.Linear(self.embedding_size, self.n_actions)
        self.apply(init_chainer_default)

    def forward(self, x):
        h = self.cnn(x)
        q = self.output(h)
        param_q = DiscreteActionValue(q)
        h_numpy = h.to("cpu").detach().numpy().copy()

        return param_q, h_numpy

class QNetworkWithValuebuffer(nn.Module):

    def __init__(self, n_actions, n_output_dim=256, ration_lambda=0.5,
                 n_neighbors=5, vbuf_capacity=2000):
        super().__init__()
        self.embedding_size = n_output_dim
        self.ration_lambda=ration_lambda
        self.n_actions = n_actions
        self.n_neighbors = n_neighbors
        self.v_buffer = ValueBuffer(capacity=vbuf_capacity, n_actions=n_actions,
                                    n_dim=n_output_dim)
        self.q_function = QNetwork(n_actions, n_output_dim)

    def forward(self, x, eva_flag=False):

        param_q, h = self.q_function(x)

        if eva_flag and len(self.v_buffer) >= self.v_buffer.capacity:
            non_param_q = self.v_buffer.get_non_param_q(h, self.n_neighbors)
            return DiscreteActionValue(self.ration_lambda*param_q.q_values + (1-self.ration_lambda)*non_param_q), h
        else:
            return param_q, h

