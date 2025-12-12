import random

import torch

from src.agents.controllers.base_controller import BaseController
from src.env.state_encoder import StateEncoder
from src.utils.config import Config


class Controller(BaseController):
    def __init__(self, in_dim: int, out_dim: int, cfg: Config, state_encoder: StateEncoder):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=cfg.hidden,
            lr=cfg.lr_ctrl,
            loss_fn=cfg.loss_fn,
            target_sync_rate=cfg.target_sync_ctrl,
            state_encoder=state_encoder,
            gamma=cfg.gamma_ctrl
        )

    def select_action(self, state, goal_idx: int, eps: float) -> int:
        """
        ε-greedy over primitive actions. Uses policy net’s argmax when exploiting.
        """
        if random.random() < eps:
            return random.randrange(self.out_dim)
        with torch.no_grad():
            controller_state = self.state_discretizer.ctrl_state(state, goal_idx)
            return self.policy(controller_state).argmax().item()

    def optimize(self, batch):
        """
        !! state and next_state still need to be converted to Tensor !!
        """
        current_q_list, target_q_list = [], []

        for state, goal_idx, action, next_state, intrinsic_reward, done_goal in batch:
            s = self.state_discretizer.ctrl_state(state, goal_idx)
            s2 = self.state_discretizer.ctrl_state(next_state, goal_idx)
            self._optimize_shared(s, s2, intrinsic_reward, done_goal, current_q_list, target_q_list, action)

        self._optimize_end(current_q_list, target_q_list)
