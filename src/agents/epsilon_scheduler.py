import numpy as np

from src.utils.config import Config


class EpsilonSchedulers:
    """
    Episode-based epsilon schedules.
    - meta ε decays with episode index
    - controller base ε decays with episode index
    - controller per-goal ε is shaped by success rate of that goal
    """

    def __init__(self, cfg: Config, goal_count: int):
        self.cfg = cfg

        # over how many episodes should eps decay; still needs to be set
        self.meta_decay_episodes = -1  # meta_decay_episodes
        self.controller_decay_episodes = -1  # controller_decay_episodes

        # per-goal success bookkeeping (for ε shaping)
        self.goal_attempts = np.zeros(goal_count, dtype=np.int32)
        self.goal_successes = np.zeros(goal_count, dtype=np.int32)

    @staticmethod
    def linear_decay(start, end, idx, horizon):
        if horizon <= 0:
            return end
        t = min(max(idx / float(horizon), 0.0), 1.0)
        return end + (start - end) * (1.0 - t)

    def set_decay_episodes(self, episodes: int):
        self.meta_decay_episodes = episodes
        self.controller_decay_episodes = episodes

    # --------- episode-based epsilons ----------
    def eps_meta(self, ep_idx: int) -> float:
        return self.linear_decay(self.cfg.eps_meta_start, self.cfg.eps_meta_end, ep_idx, self.meta_decay_episodes)

    def base_eps_ctrl(self, ep_idx: int) -> float:
        return self.linear_decay(self.cfg.eps_ctrl_start, self.cfg.eps_ctrl_end, ep_idx, self.controller_decay_episodes)

    def eps_ctrl_for_goal(self, ep_idx: int, goal_idx: int) -> float:
        """
        Controller ε for selected goal = episode-based base, shaped by success rate of that goal.
        ε_g = max(ε_end, base * (1 - success_rate_g))
        """
        base = self.base_eps_ctrl(ep_idx)
        attempts = max(1, int(self.goal_attempts[goal_idx]))
        success_rate = float(self.goal_successes[goal_idx]) / attempts
        return max(self.cfg.eps_ctrl_end, base * (1.0 - success_rate))
