from dataclasses import dataclass
from torch import nn


@dataclass
class Config:
    # discounts
    gamma_meta: float = 0.9
    gamma_ctrl: float = 0.9

    # learning rates
    lr_meta: float = 0.001
    lr_ctrl: float = 0.001

    # replay buffer
    replay_size_meta: int = 500
    replay_size_ctrl: int = 1_000
    batch_size_meta: int = 16
    batch_size_ctrl: int = 32

    # target sync (counted in environment steps)
    target_sync_meta: int = 50
    target_sync_ctrl: int = 10

    # epsilons
    eps_meta_start: float = 1.0
    eps_meta_end: float = 0.01
    eps_ctrl_start: float = 1.0
    eps_ctrl_end: float = 0.01

    # rewards
    meta_success_reward: float = 20
    meta_failure_penalty: float = -1

    controller_success_reward: float = 1
    controller_step_penalty: float = 0

    new_goal_reward: float = 2
    old_goal_penalty: float = -2
    subgoal_not_achieved_penalty: float = -0.2

    # neural network
    hidden: int = 64  # size of hidden layer
    loss_fn: nn.Module = nn.MSELoss()

    # limits
    max_steps_per_goal: int = 20
    max_episode_steps: int = 200

    # logging
    log_every_episodes: int = 2000
