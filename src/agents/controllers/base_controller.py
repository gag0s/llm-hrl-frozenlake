import torch

from torch import nn
from src.agents.dqn import DQN
from src.env.state_encoder import StateEncoder


class BaseController:
    """
    Holds everything both controllers share:
    - policy/target networks
    - optimizer
    - loss fn
    - target sync bookkeeping

    Subclasses implement:
    - select_*  (goal/action)
    - optimize  (batch-specific vector-target update)
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            state_encoder: StateEncoder,
            *,
            hidden: int,
            lr: float,
            loss_fn: nn.Module,
            target_sync_rate: int,
            gamma: float,
    ):
        self.loss_fn = loss_fn
        self.target_sync_rate = target_sync_rate
        self.gamma = gamma

        self.state_discretizer = state_encoder

        # networks
        self.policy = DQN(in_dim, out_dim, hidden)
        self.target = DQN(in_dim, out_dim, hidden)
        self.target.load_state_dict(self.policy.state_dict())

        # optimizer
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # small helper so subclasses can know output size if they need it
        self.out_dim = out_dim

    def sync_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def _optimize_shared(self, s, s2, reward, done, current_q_list, target_q_list, action, *, mask=None):
        with torch.no_grad():
            if done:
                target_scalar = torch.tensor(reward, dtype=torch.float32)
            else:
                # Double-DQN style selection with optional mask
                q_next_policy = self.policy(s2)
                if mask is not None:
                    q_next_policy = q_next_policy.clone()
                    q_next_policy[~mask] = -1e9
                next_goal = q_next_policy.argmax(dim=-1)
                q_next_target = self.target(s2)
                if mask is not None:
                    q_next_target = q_next_target.clone()
                    q_next_target[~mask] = -1e9
                next_q = q_next_target[next_goal]
                target_scalar = torch.tensor(reward, dtype=torch.float32) + self.gamma * next_q

        current_q = self.policy(s)
        target_q = self.target(s).detach().clone()
        target_q[action] = target_scalar
        current_q_list.append(current_q)
        target_q_list.append(target_q)

    def _optimize_end(self, current_q_list, target_q_list):
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

