from typing import List

import numpy as np
import torch

from pathlib import Path
from src.agents.hdqn import HDQN
from src.env.frozen_lake_env import FrozenLakeEnv
from src.llm.llm import MockLLM, ChatGPTLLM
from src.llm.prompt_box import PromptBox
from src.utils.config import Config
from src.utils.plotter import Plotter
from src.utils.visualizer import Visualizer


"""
Main training/evaluation entry point for the FrozenLake experiment.

Implements:
- HDQN training loop (meta selects subgoals, controller selects primitive actions)
- evaluation loop (collect per-subgoal stats + average reward)
- LLM feedback loop:
    prompt -> subgoals -> train -> evaluate -> feedback -> prompt -> ...
"""


def train(subgoals: List[int], fixed_variant: bool):
    env = FrozenLakeEnv(False, cfg)
    agent = HDQN(cfg, env, subgoals, fixed_variant)

    agent.schedulers.set_decay_episodes(episodes)
    rewards_per_episode = np.zeros(episodes)

    # Episode Loop
    for ep in range(episodes):

        state, _ = env.env.reset()
        ep_done = False

        visited_subgoals = set()  # extra reward for reaching new subgoals

        # Episode live
        while not ep_done:

            # === META: pick a goal ===
            meta_reward = 0
            eps_meta = agent.schedulers.eps_meta(ep)
            goal_idx = agent.meta.select_goal(state, eps_meta)

            old_state = state  # keep old state for meta transition later

            # count attempt for this goal (for controller epsilon)
            agent.schedulers.goal_attempts[goal_idx] += 1

            # === CONTROLLER: pursue goal ===
            eps_ctrl = agent.schedulers.eps_ctrl_for_goal(ep, goal_idx)
            steps_to_this_goal = 0
            subgoal_achieved = False

            # Loop until Controller reaches subgoal or timeframe is over (or ep_done)
            while not subgoal_achieved and steps_to_this_goal < cfg.max_steps_per_goal and not ep_done:
                steps_to_this_goal += 1

                # === CONTROLLER: pick action ===
                with torch.no_grad():
                    action = agent.ctrl.select_action(state, goal_idx, eps_ctrl)

                # Execute action
                new_state, extrinsic_reward, terminated, truncated, _ = env.env.step(action)
                ep_done = terminated or truncated

                if terminated:
                    if extrinsic_reward == 1:  # solved
                        meta_reward += cfg.meta_success_reward
                        rewards_per_episode[ep] = 1
                    else:  # fell in hole
                        meta_reward += cfg.meta_failure_penalty
                        rewards_per_episode[ep] = -1

                # Intrinsic reward
                if agent.subgoal_achieved(new_state, goal_idx):  # reached subgoal
                    intrinsic_reward = cfg.controller_success_reward
                    agent.schedulers.goal_successes[goal_idx] += 1
                    subgoal_achieved = True

                    if goal_idx not in visited_subgoals:  # bonus for visiting new subgoal
                        meta_reward += cfg.new_goal_reward
                        visited_subgoals.add(goal_idx)
                    else:  # penalty for visiting old subgoals
                        meta_reward += cfg.old_goal_penalty

                else:  # no subgoal reached currently
                    intrinsic_reward = cfg.controller_step_penalty
                    subgoal_achieved = False

                # Store controller transition
                agent.store_ctrl_transition(state, goal_idx, action, new_state, intrinsic_reward,
                                            subgoal_achieved or ep_done)

                # Advance state
                state = new_state
                agent.advance_step_counters()

            # Penalty for picking a subgoal and not achieving it
            if steps_to_this_goal >= cfg.max_steps_per_goal:
                meta_reward += cfg.subgoal_not_achieved_penalty

            # Store meta transition
            agent.store_meta_transition(old_state, goal_idx, state, meta_reward, ep_done)

        # === EPISODE END ===
        # Log progress
        if ep % cfg.log_every_episodes == 0:
            print(f'Episode {ep}')

        # Update controllers
        agent.update_controllers()

        # Reset fixed meta controller index
        if fixed_variant:
            agent.reset_fixed_meta()

    # === TRAINING END ===
    # Save last model
    if not fixed_variant:  # don't save fixed order meta, no DQN
        torch.save(agent.meta.policy.state_dict(), META_PATH)
    torch.save(agent.ctrl.policy.state_dict(), CTRL_PATH)

    env.env.close()
    return rewards_per_episode


def evaluate(render: bool, subgoals: List[int], fixed_variant: bool):
    env = FrozenLakeEnv(render, cfg)
    agent = HDQN(cfg, env, subgoals, fixed_variant)

    visualizer = Visualizer(env, True)

    # Load models
    if not fixed_variant:  # fixed order meta is no DQN
        agent.meta.policy.load_state_dict(torch.load(META_PATH))
        agent.meta.policy.eval()
    agent.ctrl.policy.load_state_dict(torch.load(CTRL_PATH))
    agent.ctrl.policy.eval()

    # Stats
    goal_attempts = np.zeros(env.num_states)
    goal_successes = np.zeros(env.num_states)
    average_reward_list = []

    # Episode Loop
    for ep in range(eval_episodes):

        state, _ = env.env.reset()
        ep_done = False
        reward = 0

        # Episode live
        while not ep_done:

            # === META: pick a goal ===
            with torch.no_grad():
                goal_idx = agent.meta.select_goal(state, 0)

            goal_attempts[goal_idx] += 1

            # === CONTROLLER: pursue goal ===
            steps_to_this_goal = 0

            # Loop until Controller reaches subgoal or timeframe is over (or ep_done)
            while steps_to_this_goal < cfg.max_steps_per_goal and not ep_done:
                steps_to_this_goal += 1

                # === CONTROLLER: pick action ===
                with torch.no_grad():
                    action = agent.ctrl.select_action(state, goal_idx, 0)

                # Execute action
                new_state, extrinsic_reward, terminated, truncated, _ = env.env.step(action)
                ep_done = terminated or truncated

                if terminated:
                    if extrinsic_reward == 1:  # solved
                        reward = 1
                    else:  # fell in hole
                        reward = -1

                # Subgoal Termination condition
                if agent.subgoal_achieved(new_state, goal_idx):
                    subgoal_achieved = True
                    goal_successes[goal_idx] += 1
                else:  # No subgoal reached currently
                    subgoal_achieved = False

                # Render
                if render:
                    frame = env.env.render()
                    frame = visualizer.annotate(frame, goal_idx, action)
                    if visualizer.show(frame):  # returns True if 'q' pressed
                        break

                # Advance state
                state = new_state

                # Subgoal achieved => Meta needs to choose again
                if subgoal_achieved:
                    break

        # === EPISODE END ===
        average_reward_list.append(reward)

        # Reset fixed meta controller index
        if fixed_variant:
            agent.reset_fixed_meta()

    # === TRAINING END ===
    visualizer.close()
    env.env.close()

    # if render, no need to calculate statistics
    if render:
        return 0

    # Aggregate metrics
    percentages = np.divide(
        goal_successes, goal_attempts,
        out=np.zeros_like(goal_successes, dtype=float),
        where=goal_attempts != 0
    ) * 100

    # dictionary with subgoal data
    subgoal_data = []
    for idx in subgoals:
        attempts = int(goal_attempts[idx])
        success_percentage = int(percentages[idx])
        subgoal_data.append({"index": idx, "attempts": attempts, "success_percentage": success_percentage})

    # average result
    average_result = np.average(average_reward_list)
    print(f"average agent reward: {average_result}")

    return subgoal_data, average_result


def run_once():
    fixed_variant = True  # Fixed Order Subgoals or Set of Subgoals
    subgoals = prompt_box.get_subgoals(None, None, None)

    reward_list = train(subgoals, fixed_variant)
    run_name = "Fixed Order Subgoals" if fixed_variant else "Set of Subgoals"
    plotter.add_training_run((run_name, reward_list))

    plotter.plot(PLOT_PATH)
    evaluate(True, subgoals, fixed_variant)


def run_experiment(fixed_variant: bool):
    label = "Fixed Order Subgoals" if fixed_variant else "Set of Subgoals"

    rewards_list: list[list[np.array]] = []  # list of all rewards

    for i in range(training_runs):
        print(f"{label} Run {i + 1} / {training_runs}")

        iteration_rewards: list[np.array] = []  # list of rewards of all iterations

        # With each iteration, LLM proposes new subgoals
        subgoals = None
        subgoal_data = None
        avg_reward = None

        for iteration in range(llm_iterations):
            print(f"Iteration {iteration + 1} / {llm_iterations}")

            # 1) LLM proposes new subgoals
            subgoals = prompt_box.get_subgoals(subgoals, subgoal_data, avg_reward)

            # 2) Train agent
            rewards = train(subgoals, fixed_variant=fixed_variant)
            iteration_rewards.append(rewards)

            # 3) Collect feedback stats for next prompt
            subgoal_data, avg_reward = evaluate(render=False, subgoals=subgoals, fixed_variant=fixed_variant)

        rewards_list.append(iteration_rewards)

    # Compute averages for each iteration and plot them
    avg_per_iteration = []
    for it in range(llm_iterations):
        avg_it = np.average([iteration_rewards[it] for iteration_rewards in rewards_list], axis=0)
        avg_per_iteration.append(avg_it)

    runs = [(f"{label} Iteration {i + 1}", avg_per_iteration[i]) for i in range(llm_iterations)]
    plotter.add_many(runs)


# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

META_PATH = MODELS_DIR / "hdqn_meta.pt"
CTRL_PATH = MODELS_DIR / "hdqn_ctrl.pt"
PLOT_PATH = PLOTS_DIR / "training_rewards.png"

# =============

cfg = Config()
llm = MockLLM()  # or ChatGPTLLM()
prompt_box = PromptBox(llm, FrozenLakeEnv.FL_MAP)
plotter = Plotter()

# Experiment Settings
episodes = 10_000
eval_episodes = 100
llm_iterations = 2
training_runs = 1

if __name__ == "__main__":
    # === EXPERIMENT ===

    # === Set of Subgoals ===
    run_experiment(fixed_variant=False)

    # === Fixed Order Subgoals ===
    run_experiment(fixed_variant=True)

    # === PLOT ===
    plotter.plot(save_path=PLOT_PATH)
