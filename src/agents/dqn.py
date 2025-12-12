import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, in_states, out_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
