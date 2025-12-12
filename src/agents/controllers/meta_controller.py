import random

import torch

from src.agents.controllers.base_controller import BaseController
from src.env.state_encoder import StateEncoder
from src.utils.config import Config


class MetaController(BaseController):
    def __init__(self, in_dim: int, out_dim: int, cfg: Config, state_encoder: StateEncoder, subgoals: list[int]):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=cfg.hidden,
            lr=cfg.lr_meta,
            loss_fn=cfg.loss_fn,
            target_sync_rate=cfg.target_sync_meta,
            state_encoder=state_encoder,
            gamma=cfg.gamma_meta
        )

        self.allowed_mask = None
        self.set_subgoals(subgoals)

    def set_subgoals(self, allowed_indices: list[int]):
        """allowed_indices are zero-based goal IDs (e.g., [2,5,7])."""
        mask = torch.zeros(self.out_dim, dtype=torch.bool)
        for i in allowed_indices:
            if 0 <= i < self.out_dim:
                mask[i] = True
        # if nothing valid was provided, fall back to 'all allowed'
        if mask.sum() == 0:
            mask[:] = True
        self.allowed_mask = mask

    def _allowed_indices_list(self):
        if self.allowed_mask is None:
            # default: all goals allowed
            return list(range(self.out_dim))
        return torch.nonzero(self.allowed_mask, as_tuple=False).view(-1).tolist()

    def select_goal(self, state, eps: float) -> int:
        """
        ε-greedy over *allowed* goals only.
        """
        allowed_idxs = self._allowed_indices_list()
        if len(allowed_idxs) == 0:
            return 0  # safe fallback

        if random.random() < eps:
            return random.choice(allowed_idxs)

        with torch.no_grad():
            meta_state = self.state_discretizer.meta_state(state)
            q = self.policy(meta_state)
            if self.allowed_mask is not None:
                # mask out disallowed with large negative
                q = q.clone()
                q[~self.allowed_mask] = -1e9
            return q.argmax().item()

    def optimize(self, batch):
        """
        !! meta_state and next_meta_state still need to be converted to Tensor !!
        """
        current_q_list, target_q_list = [], []

        mask = self.allowed_mask

        for meta_state, subgoal, next_meta_state, reward, done in batch:
            s = self.state_discretizer.meta_state(meta_state)
            s2 = self.state_discretizer.meta_state(next_meta_state)
            self._optimize_shared(s, s2, reward, done, current_q_list, target_q_list, subgoal, mask=mask)  # todo

        self._optimize_end(current_q_list, target_q_list)
