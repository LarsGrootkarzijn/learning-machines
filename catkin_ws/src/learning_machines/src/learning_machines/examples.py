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

def example1(rob: IRobobo, runs=10, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1):
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
    plt.plot(avg_rewards, color="blue", marker='o')
    plt.title("Average Reward Per Episode Across Runs")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/root/results/qlearning_ir_all_sensors.png")
    plt.show()


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
