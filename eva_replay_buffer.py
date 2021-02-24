import collections
import pickle
from typing import List, Optional

import numpy as np
import torch
from pfrl.collections.random_access_queue import RandomAccessQueue
from pfrl import replay_buffer

from lsm_knn_buffer import LSMKNNBuffer as lkb


class EVAReplayBuffer(replay_buffer.AbstractReplayBuffer):
    """Experience Replay Buffer

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    In addition to the normal replay buffer, storing features.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    # Implements AbstractReplayBuffer.capacity
    capacity: Optional[int] = None

    def __init__(self, capacity, n_dim=256, n_neighbors=5, num_steps=1):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.h_memory = lkb(capacity=capacity, n_dim=n_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.current_embeddings = torch.empty(0, n_dim, device=self.device, dtype=torch.float32)
        # self.current_embeddings = []
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )
        self.n_neighbors = n_neighbors

    def append(
        self,
        state,
        action,
        reward,
        feature : torch.tensor,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            feature=feature,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                embeddings = [m['feature'] for m in last_n_transitions]
                self.update_feature_arr(embeddings)
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))
                embeddings = [m['feature'] for m in last_n_transitions]
                self.update_feature_arr(embeddings)
        assert len(self.h_memory) == len(self)

    def stop_current_episode(self, env_id=0):
        last_n_transitions = self.last_n_transitions[env_id]
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(last_n_transitions) < self.num_steps:
            self.memory.append(list(last_n_transitions))
            embeddings = [m['feature'] for m in last_n_transitions]
            self.update_feature_arr(embeddings)
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del last_n_transitions[0]
        while last_n_transitions:
            self.memory.append(list(last_n_transitions))
            embeddings = [m['feature'] for m in last_n_transitions]
            self.update_feature_arr(embeddings)
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0
        assert len(self.h_memory) == len(self)

    def sample(self, num_experiences):
        assert len(self.memory) >= num_experiences
        return self.memory.sample(num_experiences)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)
        if isinstance(self.memory, collections.deque):
            # Load v0.2
            self.memory = RandomAccessQueue(self.memory, maxlen=self.memory.maxlen)

    def update_feature_arr(self, embeddings : List[np.ndarray]):
        if len(embeddings) > 0:
            # list -> numpy
            added = np.asarray(embeddings, dtype=np.float32)
            # numpy -> Tensor
            added = torch.from_numpy(added)
            self.h_memory.append(added)

        assert len(self.h_memory) == len(self)

    def lookup(self, target_h, max_len):

        assert len(self.h_memory) == len(self)

        target_h = torch.from_numpy(target_h).clone()
        start_indices = self.h_memory.search(target_h.reshape(1,-1), self.n_neighbors)

        trajectory_list = []
        for start_index in start_indices:
            trajectory = []
            for sub_sequence in range(max_len):
                step = self.memory[start_index + sub_sequence]
                trajectory.append(step[0])
                if step[0]["is_state_terminal"]:
                    break
                if (start_index + sub_sequence) == (len(self.memory) - 1):
                    break
            trajectory_list.append(trajectory)

        return trajectory_list