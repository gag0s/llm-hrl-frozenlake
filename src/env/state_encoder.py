import torch

from src.env.frozen_lake_env import FrozenLakeEnv


class StateEncoder:
    """Encodes discrete environment states (and goals) as one-hot PyTorch tensors for the meta-controller and
    controller."""

    def __init__(self, env: FrozenLakeEnv):
        self.env = env

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        input_tensor = torch.zeros(self.env.num_states)
        input_tensor[state] = 1
        return input_tensor

    def meta_state(self, state) -> torch.Tensor:
        return self._state_to_tensor(state)

    def ctrl_state(self, state: int, goal_idx: int) -> torch.Tensor:
        state_tensor = self._state_to_tensor(state)
        subgoal_tensor = self._state_to_tensor(goal_idx)
        return torch.cat([state_tensor, subgoal_tensor], dim=0)
