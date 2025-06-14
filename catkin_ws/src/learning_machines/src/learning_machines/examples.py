import cv2

try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
SENSOR_NAMES = [
    "BackL", "BackR", "FrontL", "FrontR",
    "FrontC", "FrontRR", "BackC", "FrontLL"
]

ACTIONS = [
    (50, 50),   # move forward
    (50, -50),  # turn right
    (-50, 50),  # turn left
]

NUM_ACTIONS = len(ACTIONS)
OBSTACLE_THRESHOLD = 30

def get_state(irs):
    front_sensors = [irs[2], irs[3], irs[4], irs[5], irs[7]]
    back_sensors = [irs[0], irs[1], irs[6]]
    discrete_front = [1 if val > OBSTACLE_THRESHOLD else 0 for val in front_sensors]
    discrete_back = [1 if val > OBSTACLE_THRESHOLD else 0 for val in back_sensors]
    return tuple(discrete_front + discrete_back)

def get_reward(irs, action_idx):
    front_c = irs[4]   # FrontC
    front_ll = irs[7]  # FrontLL
    front_rr = irs[5]  # FrontRR
    front_l = irs[2]   # FrontL
    front_r = irs[3]   # FrontR
    back_l = irs[0]    # BackL
    back_r = irs[1]    # BackR
    back_c = irs[6]    # BackC

    reward = 0.0

    # Encourage moving forward if path is clear
    if front_c < OBSTACLE_THRESHOLD:
        if action_idx == 0:  # move forward
            reward += 1.0
        else:  # penalize turning if going straight was possible
            reward -= 1.0
    else:
        reward -= 5.0  # collision ahead

    # Encourage turning away from nearby obstacles
    if front_l > OBSTACLE_THRESHOLD and front_r < OBSTACLE_THRESHOLD and action_idx == 1:
        reward += 0.5
    elif front_r > OBSTACLE_THRESHOLD and front_l < OBSTACLE_THRESHOLD and action_idx == 2:
        reward += 0.5

    # Extra bonus if surrounded at back too and makes a turn
    if back_l > OBSTACLE_THRESHOLD and front_r > OBSTACLE_THRESHOLD and action_idx == 1:
        reward += 1.0
    elif back_r > OBSTACLE_THRESHOLD and front_l > OBSTACLE_THRESHOLD and action_idx == 2:
        reward += 1.0

    return reward

def run_single_trial(rob: IRobobo, runs=10, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {}
    trial_rewards = []      # Will store average reward per run
    trial_violations = []   # Will store total violations per run
    best_q_table = None
    best_avg_reward = float('-inf')

    for run in range(runs):
        print(f"  Run {run + 1}/{runs}")
        rob.play_simulation()
        initial_pos = rob.get_position()
        initial_ori = rob.get_orientation()

        total_run_reward = 0
        total_run_violations = 0

        for ep in range(episodes):
            irs = rob.read_irs()
            state = get_state(irs)
            total_reward = 0
            violation_count = 0

            for step in range(30):
                if random.random() < epsilon or state not in q_table:
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action_idx = np.argmax(q_table[state])

                left_speed, right_speed = ACTIONS[action_idx]
                rob.move_blocking(left_speed, right_speed, 100)

                next_irs = rob.read_irs()
                reward = get_reward(next_irs, action_idx)
                total_reward += reward

                # Count violation if FrontC > 50
                if next_irs[4] > 100:
                    violation_count += 1

                next_state = get_state(next_irs)

                if state not in q_table:
                    q_table[state] = [0.0] * NUM_ACTIONS
                if next_state not in q_table:
                    q_table[next_state] = [0.0] * NUM_ACTIONS

                old_value = q_table[state][action_idx]
                next_max = max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state][action_idx] = new_value

                state = next_state

            total_run_reward += total_reward
            total_run_violations += violation_count

        rob.set_position(initial_pos, initial_ori)
        rob.reset_wheels()
        rob.stop_simulation()

        # Store per-run aggregates
        trial_rewards.append(total_run_reward)
        trial_violations.append(total_run_violations)

        avg_reward_this_run = total_run_reward / episodes
        if avg_reward_this_run > best_avg_reward:
            best_avg_reward = avg_reward_this_run
            best_q_table = q_table.copy()

        # Save best q_table to file
        with open('/root/results/best_q_table.pkl', 'wb') as f:
            pickle.dump(best_q_table, f)

    return trial_rewards, trial_violations


def example1(rob: IRobobo, trials=15, runs=10, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1):
    all_avg_rewards = []
    all_avg_violations = []

    for trial in range(trials):
        print(f"\n=== Trial {trial + 1}/{trials} ===")
        avg_rewards, avg_violations = run_single_trial(
            rob, runs=runs, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        all_avg_rewards.append(avg_rewards)
        all_avg_violations.append(avg_violations)

    final_avg_rewards = np.mean(all_avg_rewards, axis=0)
    final_avg_violations = np.mean(all_avg_violations, axis=0)

    # Optional: std dev for shading
    std_rewards = np.std(all_avg_rewards, axis=0)
    std_violations = np.std(all_avg_violations, axis=0)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(final_avg_rewards, label="Avg Reward", color="blue", marker='o')
    plt.fill_between(range(episodes), final_avg_rewards - std_rewards, final_avg_rewards + std_rewards,
                     color="blue", alpha=0.2, label="±1 std")
    plt.title("Average Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Violation plot
    plt.subplot(1, 2, 2)
    plt.plot(final_avg_violations, label="Avg FrontC > 100", color="red", marker='x')
    plt.fill_between(range(episodes), final_avg_violations - std_violations, final_avg_violations + std_violations,
                     color="red", alpha=0.2, label="±1 std")
    plt.title("Avg Violation Count Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    import os
    #os.makedirs("results", exist_ok=True)
    plt.savefig("/root/results/qlearning_final.png")
    plt.show()

def example2(rob: IRobobo, q_table_path='/root/results/best_q_table.pkl', episodes=10):
    import pickle

    with open(q_table_path, 'rb') as f:
        q_table = pickle.load(f)

    rob.play_simulation()
    initial_pos = rob.get_position()
    initial_ori = rob.get_orientation()

    for ep in range(episodes):
        irs = rob.read_irs()
        state = get_state(irs)
        print(f"Episode {ep + 1}")

        for step in range(30):
            if state in q_table:
                action_idx = np.argmax(q_table[state])
            else:
                action_idx = random.randint(0, NUM_ACTIONS - 1)

            left_speed, right_speed = ACTIONS[action_idx]
            rob.move_blocking(left_speed, right_speed, 100)

            next_irs = rob.read_irs()
            next_state = get_state(next_irs)
            state = next_state

    rob.set_position(initial_pos, initial_ori)
    rob.reset_wheels()
    rob.stop_simulation()

def example2(rob: IRobobo, runs=10, episodes=50, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {}
    results = []

    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        rob.play_simulation()
        initial_pos = rob.get_position()
        initial_ori = rob.get_orientation()

        episode_rewards = []

        for ep in range(episodes):
            irs = rob.read_irs()
            state = get_state(irs)
            total_reward = 0

            for step in range(30):
                if random.random() < epsilon or state not in q_table:
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action_idx = np.argmax(q_table[state])

                left_speed, right_speed = ACTIONS[action_idx]
                rob.move_blocking(left_speed, right_speed, 100)

                next_irs = rob.read_irs()
                reward = get_reward(next_irs, action_idx)
                total_reward += reward

                next_state = get_state(next_irs)

                if state not in q_table:
                    q_table[state] = [0.0] * NUM_ACTIONS
                if next_state not in q_table:
                    q_table[next_state] = [0.0] * NUM_ACTIONS

                old_value = q_table[state][action_idx]
                next_max = max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state][action_idx] = new_value

                state = next_state

            episode_rewards.append(total_reward)

        results.append(episode_rewards)
        rob.set_position(initial_pos, initial_ori)
        rob.reset_wheels()
        rob.stop_simulation()

    avg_rewards = np.mean(results, axis=0)
    plt.plot(avg_rewards, color="green", marker='x')
    plt.title("Example 2: Average Reward Per Episode Across Runs")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/root/results/qlearning_ir_example2.png")
    plt.show()
