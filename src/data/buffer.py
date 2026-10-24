import numpy as np
import torch


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""

    def __init__(self, buffer_size: int, n_seeds: int, state_dim: int, batch_size: int, device: torch.device) -> None:
        self.buffer_size = buffer_size
        self.n_seeds = n_seeds
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.device = device

        self.states = np.zeros((buffer_size, n_seeds, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_seeds), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_seeds), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, n_seeds, state_dim), dtype=np.float32)
        self.terminals = np.zeros((buffer_size, n_seeds), dtype=np.int64)

        self.idx = 0
        self.current_size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
            terminal: np.ndarray) -> None:
        if self.current_size < self.buffer_size:
            self.current_size += 1

        self.states[self.idx, :, :] = state
        self.actions[self.idx, :] = action
        self.rewards[self.idx, :] = reward
        self.next_states[self.idx, :, :] = next_state
        self.terminals[self.idx, :] = terminal

        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idxs = np.random.permutation(self.current_size)[:self.batch_size]

        states = torch.from_numpy(self.states[sample_idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[sample_idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[sample_idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[sample_idxs]).to(self.device)
        terminals = torch.from_numpy(self.terminals[sample_idxs]).to(self.device)

        return states, actions, rewards, next_states, terminals