import gymnasium as gym

from src.utils.config import Config


class FrozenLakeEnv:
    """Thin wrapper around Gymnasium FrozenLake-v1 to expose sizes and map."""

    FL_MAP = [
        "SFFFHFFF",
        "FHFFFFHF",
        "FHFHFFFF",
        "FHHFHHHF",
        "FHFFFFFH",
        "FHFHHFFF",
        "FHFFFFHF",
        "FFFHFFHG",
    ]

    def __init__(self, render: bool, cfg: Config):
        self.cfg = cfg

        # Deterministic FrozenLake (is_slippery=False) with custom 8x8 map.
        self.env = gym.make("FrozenLake-v1", render_mode="rgb_array" if render else None, is_slippery=False,
                            desc=FrozenLakeEnv.FL_MAP, map_name="8x8", max_episode_steps=cfg.max_episode_steps)

        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
