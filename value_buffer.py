import torch

from lsm_knn_buffer import LSMKNNBuffer as lkb

class ValueBuffer:

    def __init__(self, capacity:int, n_actions:int, n_dim:int, dtype=torch.float32):
        self.capacity = capacity
        self.n_dim = n_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_memory = torch.empty(0, n_actions, device=self.device, dtype=torch.float32)
        self.h_memory = lkb(capacity=capacity, n_dim=n_dim)

    def store(self, features, q_values):
        self.h_memory.append(features)
        self.q_memory = torch.cat((self.q_memory, q_values), dim=0)

        if self.q_memory.size()[0] > self.capacity:
            self.q_memory = self.q_memory[-self.capacity:]

    def get_non_param_q(self, current_h, n_index):

        assert len(self.h_memory) == self.q_memory.size()[0]

        current_h = torch.from_numpy(current_h).clone()
        indices = self.h_memory.search(current_h, n_index)
        non_q_values = self.q_memory[indices]

        return torch.mean(non_q_values, dim=0).reshape(1, -1)

    def __len__(self):
        return self.q_memory.size()[0]