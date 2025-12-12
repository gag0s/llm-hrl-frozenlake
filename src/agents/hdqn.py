from src.agents.controllers.controller import Controller
from src.agents.controllers.fixed_meta_controller import FixedMetaController
from src.agents.controllers.meta_controller import MetaController
from src.agents.epsilon_scheduler import EpsilonSchedulers
from src.agents.replay_memory import ReplayMemory
from src.env.frozen_lake_env import FrozenLakeEnv
from src.env.state_encoder import StateEncoder
from src.utils.config import Config


class HDQN:
    """
    Hierarchical Deep Q-Network (HDQN) agent for FrozenLake.

    Structure:
    - Meta-controller selects a subgoal (state index 0..63)
      - either learned via DQN (set-of-subgoals variant)
      - or a fixed sequential list (fixed-order variant)
    - Controller selects primitive actions to reach the chosen subgoal.

    This class wires both levels together, manages replay buffers and performs updates.
    """
    def __init__(self, cfg: Config, env: FrozenLakeEnv, subgoals: list[int], fixed_variant: bool):
        self.cfg = cfg
        self.env = env
        self.fixed_variant = fixed_variant

        self.state_encoder = StateEncoder(self.env)
        self.schedulers = EpsilonSchedulers(cfg, self.env.num_states)

        # controllers
        self.meta = FixedMetaController(subgoals) if fixed_variant else MetaController(self.env.num_states, self.env.num_states, cfg, self.state_encoder, subgoals)
        self.ctrl = Controller(self.env.num_states * 2, self.env.num_actions, cfg, self.state_encoder)

        # buffers
        # Meta replay buffer is not used in fixed-order variant because no meta DQN is trained.
        self.memory_meta = None if fixed_variant else ReplayMemory(self.cfg.replay_size_meta)
        self.memory_ctrl = ReplayMemory(self.cfg.replay_size_ctrl)

        # step counters (for target syncing)
        self.step_count_meta = 0
        self.step_count_ctrl = 0

    def advance_step_counters(self):
        self.step_count_meta += 1
        self.step_count_ctrl += 1

    def subgoal_achieved(self, state: int, goal_idx: int) -> bool:
        achieved = state == goal_idx

        # move onto next subgoal for fixed order meta
        if achieved and self.fixed_variant:
            self.meta.subgoal_done()

        return achieved

    def store_meta_transition(self, old_state, goal_idx, new_state, meta_reward, terminated):
        if not self.fixed_variant:  # only store if not fixed order meta controller
            self.memory_meta.append((old_state, goal_idx, new_state, meta_reward, terminated))

    def store_ctrl_transition(self, state, goal_idx, action, new_state, intrinsic_reward, terminated):
        self.memory_ctrl.append((state, goal_idx, action, new_state, intrinsic_reward, terminated))

    def update_controllers(self):
        # Update Meta Controller (if not fixed order meta controller)
        if not self.fixed_variant and len(self.memory_meta) > self.cfg.batch_size_meta:
            batch = self.memory_meta.sample(self.cfg.batch_size_meta)
            self.meta.optimize(batch)

            # Sync target network
            if self.step_count_meta > self.cfg.target_sync_meta:
                self.meta.sync_target()
                self.step_count_meta = 0

        # Update Controller
        if len(self.memory_ctrl) > self.cfg.batch_size_ctrl:
            batch = self.memory_ctrl.sample(self.cfg.batch_size_ctrl)
            self.ctrl.optimize(batch)

            # Sync target network
            if self.step_count_ctrl > self.cfg.target_sync_ctrl:
                self.ctrl.sync_target()
                self.step_count_ctrl = 0

    def reset_fixed_meta(self):
        if self.fixed_variant:
            self.meta.reset_counter()
