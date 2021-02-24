import torch

class LSMKNNBuffer:

    def __init__(self, capacity:int, n_dim:int, dtype=torch.float32):
        self.capacity = capacity
        self.n_dim = n_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        assert n_dim is not None
        assert capacity is not None

        self.memory = torch.empty(0, n_dim, device=self.device, dtype=self.dtype)


    def append(self, add_arr:torch.Tensor):

        assert add_arr.size()[1] == self.n_dim

        add_arr = add_arr.to(self.device)
        self.memory = torch.cat((self.memory, add_arr), dim=0)
        self._shave()

    def _shave(self):

        if self.memory.size()[0] <= self.capacity:
            pass

        self.memory = self.memory[-self.capacity:]

    def search(self, target:torch.Tensor, n_index:int):

        assert target.size() == torch.Size([1, self.n_dim])

        target = target.to(self.device)
        index_arr = torch.argsort(torch.sum((self.memory-target)**2, dim=1))[:n_index]

        return index_arr.tolist()

    def __len__(self):
        return self.memory.size()[0]
