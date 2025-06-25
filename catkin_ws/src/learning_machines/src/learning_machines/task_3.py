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
    (100, 100, 250),   # move forward small
    (100, 100, 500),   # move forward medium
    (100, 100, 750),   # move forward fast
    (30, -30, 250),    # turn right small
    (30, -30, 500),    # turn right medium
    (30, -30, 750),    # turn right fast
    (-30, 30, 250),    # turn left small
    (-30, 30, 500),    # turn left medium
    (-30, 30, 750),    # turn left fast
]
STRAIGHT_ACTION_INDEX = [0, 1, 2]  # update to your actual index for "go straight"

NUM_ACTIONS = len(ACTIONS)
OBSTACLE_THRESHOLD = 65
OPPOSITE_ACTIONS = {
    0: None, 
    1: None,
    2: None,
    3: 6,
    4: 7, 
    5: 8,
    6: 3,
    7: 4,
    8: 5
}

def downsample_mask(mask, grid_size=(3, 1)):
    h, w = mask.shape
    gh, gw = grid_size
    cell_h, cell_w = h // gh, w // gw
    
    grid = np.zeros((gh, gw), dtype=np.uint8)

    for i in range(gh):
        for j in range(gw):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = mask[y1:y2, x1:x2]
            if np.any(cell):  # if any pixel in the cell is non-zero 
                grid[i, j] = 1

    return grid

def to_list(grid):
    return list(grid.flatten())

def get_state(irs, red_grid, green_grid):
    front_sensors = [irs[2], irs[3], irs[4], irs[7]]
    back_sensors = [irs[0], irs[1], irs[6]]
    discrete_front = [1 if val > OBSTACLE_THRESHOLD else 0 for val in front_sensors]
    discrete_back = [1 if val > OBSTACLE_THRESHOLD else 0 for val in back_sensors]
    return tuple(discrete_front + discrete_back + [2 * a + b for a, b in zip(to_list(red_grid), to_list(green_grid))]
)

def get_reward(hit_wall, red_grid, green_grid, action_idx, previous_action_idx):
    reward = 0.0

    # If box is in attachment
    if np.any(red_grid[-1] == 1) and np.all(red_grid[:-1] == 0):
        # If robot sees green area
        if np.any(green_grid[:] == 1):
            # reward forward
            if action_idx < 3:
                reward += 100.0
            # penalise turning
            else: 
                reward -= 50
        # If robot doesnt see green area
        else:
            # reward turning
            if action_idx >= 3:
                reward += 50
            # penalise forward
            else:
                reward -= 50
    # If box is not in attachment
    else:
        # If robot sees box in the center
        if np.any(red_grid[ : ,1] == 1):
            # reward forward
            if action_idx < 3:
                reward += 100
            # penalise turning
            else:
                reward -= 50
        # else
        else:
            # reward turning
            if action_idx >= 3:
                reward += 50
            # penalise forward
            else:
                reward -= 50
    
    # Punish for hitting walls if nothing is on sight
    reward -= hit_wall*100
    
    # Punish for performing opposite actions 
    if previous_action_idx and OPPOSITE_ACTIONS[action_idx] == previous_action_idx:
        reward -= 50

    return reward

#this function runs a single trial of Qlearning. in this case 10 runs, each run containing 10 episodes
def task_3_run_single_trial(rob, runs=20, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1, retrain=True):
    """One trial = several independent runs, each with several episodes."""
    
    q_table_path=Path('/root/results/task_3_best_q_table.pkl')
    if retrain:
        with q_table_path.open("rb") as f:
            q_table = pickle.load(f)
    else:
        q_table = {}
    trial_rewards     = []   # sum of rewards per run
    trial_violations  = []   # total violations per run
    straight_stats    = []   # (straight_attempts, straight_hits) per run

    best_q_table   = None
    best_avg_reward = float("-inf")
    for run in range(runs):
        print(f"  Run {run + 1}/{runs}")
        rob.play_simulation()
        rob.set_phone_tilt_blocking(109, 30)

        total_run_reward      = 0
        total_run_violations  = 0
        run_straight_attempts = 0
        run_straight_hits     = 0
        previous_action_idx = None
        hit_wall = 0
        red_grid = np.zeros((3,3), dtype=int)
        green_grid = np.zeros((3,3), dtype=int)
        for ep in range(episodes):
            irs   = rob.read_irs()
            state = get_state(irs, red_grid, green_grid)
            for step in range(30):
                # ε-greedy
                if random.random() < epsilon or state not in q_table:
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action_idx = int(np.argmax(q_table[state]))

                left_speed, right_speed, millis = ACTIONS[action_idx]
                
                rob.move_blocking(left_speed, right_speed, millis)

                # Original frame
                img = cv2.flip(rob.read_image_front(), -1)

                # Convert Image to HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # HSV range for green
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([179, 255, 255])

                # Mask, filter, grid, contours
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)
                red_matrix = (mask > 0).astype(np.uint8)
                red_grid = downsample_mask(red_matrix, (3, 3))

                # HSV range for green
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])

                # Mask, filter, grid, contours
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                green_matrix = (mask_green > 0).astype(np.uint8)
                green_grid = downsample_mask(green_matrix, (3, 3))

                cv2.imwrite("/root/results/red_block.png", img)
                cv2.imwrite("/root/results/red_block_mask.png", mask)
                next_irs = rob.read_irs()
                # Checks if it hits the wall
                if any(x > 65 for x in [next_irs[2], next_irs[3], next_irs[4], next_irs[5], next_irs[7]]):
                    total_run_violations += 1
                    hit_wall = 1 

                # straight-action bookkeeping
                if action_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                    run_straight_attempts += 1
                    if next_irs[4] > 65:
                        run_straight_hits += 1
                
                reward = get_reward(hit_wall, red_grid, green_grid, action_idx, previous_action_idx)
                total_run_reward += reward
                print(f"Reward in run {run}, episode {ep}, step {step}: {reward}, hit_wall: {hit_wall}, Red Grid: {red_grid}, Green Grid: {green_grid}")
                # Q-update ---------------------------------------------------
                next_state = get_state(next_irs,red_grid, green_grid)
                q_table.setdefault(state,      [0.0] * NUM_ACTIONS)
                q_table.setdefault(next_state, [0.0] * NUM_ACTIONS)

                old_q   = q_table[state][action_idx]
                next_max = max(q_table[next_state])
                q_table[state][action_idx] = (1 - alpha) * old_q + alpha * (reward + gamma * next_max)

                state = next_state
                previous_action_idx = action_idx
                hit_wall = 0

                if rob.base_detects_food():
                    rob.stop_simulation()
                    rob.play_simulation()
                    rob.set_phone_tilt_blocking(109, 30)


        rob.stop_simulation()
        hit_wall = 0
        # per-run aggregates
        trial_rewards   .append(total_run_reward)
        trial_violations.append(total_run_violations)
        straight_stats  .append((run_straight_attempts, run_straight_hits))

        avg_reward_this_run = total_run_reward / episodes
        if avg_reward_this_run > best_avg_reward:
            best_avg_reward = avg_reward_this_run
            best_q_table    = q_table.copy()

    # persist the best Q-table found in this trial
    with open("/root/results/task_3_best_q_table.pkl", "wb") as f:
        pickle.dump(best_q_table, f)

    return trial_rewards, trial_violations, straight_stats

#this function runs an experiment. multiple trials with in each trial multiple runs, 10 episodes again. for plotting and stat significance trials should be 30 usually
def task_3_run_experiment(rob,trials=5,runs=10,episodes=10,alpha=0.1,gamma=0.9,epsilon=0.1):
    # these hold one array per trial; each array length == runs
    all_run_rewards    = []
    all_run_violations = []
    all_run_attempts   = []

    for trial in range(trials):
        print(f"\n=== Trial {trial + 1}/{trials} ===")
        run_rewards, run_violations, straight_stats = task_3_run_single_trial(
            rob, runs, episodes, alpha, gamma, epsilon
        )

        # unpack (attempts, hits) -> attempts
        attempts, _ = zip(*straight_stats)

        all_run_rewards   .append(run_rewards)
        all_run_violations.append(run_violations)
        all_run_attempts  .append(attempts)

    # ---------- averages over all trials, per run --------------------------
    mean_rewards    = np.mean(all_run_rewards,    axis=0)
    mean_violations = np.mean(all_run_violations, axis=0)
    mean_attempts   = np.mean(all_run_attempts,   axis=0)

    # standard deviations (for shading if desired)
    sd_rewards    = np.std(all_run_rewards,    axis=0)
    sd_violations = np.std(all_run_violations, axis=0)

    # ---------------- PLOTS -------------------------------------------------
    # 1) reward & violation vs. episode (aggregated over runs)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mean_rewards, label="Avg reward / run", marker="o")
    plt.fill_between(range(runs), mean_rewards - sd_rewards, mean_rewards + sd_rewards, alpha=0.2)
    plt.xlabel("Run"); plt.ylabel("Total reward"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mean_violations, label="Avg violations / run", marker="x", color="red")
    plt.fill_between(range(runs), mean_violations - sd_violations, mean_violations + sd_violations,
                     alpha=0.2, color="red")
    plt.xlabel("Run"); plt.ylabel("Violation count"); plt.grid(True); plt.legend()

    plt.suptitle("Reward and Violation per Run (averaged over trials)")
    plt.tight_layout()
    plt.savefig("/root/results/qlearning_final.png")
    plt.show()

       # 2) straight attempts minus violations per run  ------------------------
    all_differences = np.array(all_run_attempts) - np.array(all_run_violations)
    mean_diff = np.mean(all_differences, axis=0)
    std_diff  = np.std(all_differences, axis=0)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_diff, label="Straight Attempts − Violations", color="blue", marker='s')
    plt.fill_between(range(runs), mean_diff - std_diff, mean_diff + std_diff,
                     color="blue", alpha=0.2, label="±1 std")
    plt.axhline(0, linestyle="--", linewidth=0.8, color="gray")
    plt.title("Net Straight Attempts Per Run")
    plt.xlabel("Run")
    plt.ylabel("Attempts − Violations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/root/results/qlearning_straight_minus_violations.png")
    plt.show()


import cv2
import numpy as np
import pickle
import random
from pathlib import Path

# ------------------------------------------------------------------
# CONSTANTS you already have elsewhere
# ------------------------------------------------------------------
NUM_ACTIONS           = len(ACTIONS)        # e.g. 5
STRAIGHT_ACTION_INDEX = ...                 # keep yours
# ------------------------------------------------------------------

ACTIONS_REAL_LIFE = [
    (100, 100, 750),   # move forward small
    (100, 100, 750),   # move forward medium
    (100, 100, 750),   # move forward fast
    (30, -30, 750),    # turn right small
    (30, -30, 750),    # turn right medium
    (30, -30, 750),    # turn right fast
    (-30, 30, 750),    # turn left small
    (-30, 30, 750),    # turn left medium
    (-30, 30, 750),    # turn left fast
]

def task_3_real_life(
    rob,                                # IRobobo or SimulationRobobo
    q_table_path='/root/results/task_2_best_q_table.pkl',
    episodes=10,
    min_area=100,                       # ignore blobs smaller than this
    save_images=False,                  # set True to store annotated frames
    image_dir=Path("/root/results/real_life_frames"),
):
    """
    Execute the best (frozen) Q-table without updating it.
    Adds camera processing to count green blocks each step.
    """

    # ------------------------------------------------------------------
    # 1) Load trained Q-table
    # ------------------------------------------------------------------
    q_table_path = Path(q_table_path)
    if not q_table_path.exists():
        raise FileNotFoundError(f"No Q-table found at {q_table_path}")

    with q_table_path.open("rb") as f:
        q_table = pickle.load(f)

    # ------------------------------------------------------------------
    # 2) If running in simulation, remember pose so we can reset later
    # ------------------------------------------------------------------
    sim_mode = isinstance(rob, SimulationRobobo)
    if sim_mode:
        rob.play_simulation()
        initial_pos, initial_ori = rob.get_position(), rob.get_orientation()

    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    rob.set_phone_tilt_blocking(95, 30)
    n_blocks = 0
    grid = list(np.zeros(3))
    # ------------------------------------------------------------------
    # 3) Main episode loop
    # ------------------------------------------------------------------
    for ep in range(episodes):
        irs   = rob.read_irs()
        state = get_state(irs, n_blocks, grid)
        print(f"\nEpisode {ep + 1}/{episodes}")

        for step in range(30):
            # Greedy policy (no ε-exploration)
            if state in q_table:
                action_idx = int(np.argmax(q_table[state]))
            else:                       # unknown state fallback
                action_idx = random.randrange(NUM_ACTIONS)

            left_speed, right_speed, millis = ACTIONS_REAL_LIFE[action_idx]
            rob.move(left_speed, right_speed, millis)

            # -------------- CAMERA: detect green blocks -----------------
            img  = rob.read_image_front()
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(
                hsv,
                np.array([35, 40, 40]),   # lower_green
                np.array([85, 255, 255])  # upper_green
            )
            green_matrix = (mask > 0).astype(np.uint8)
            grid = downsample_mask(green_matrix, (3, 1))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            min_area = 100
            count = 0
            for cnt in contours:
                area_box = cv2.contourArea(cnt)
                if area_box > min_area:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                    count += 1
            
            cv2.imwrite("/root/results/green_blocks_detected.png", img)
            blocks = [
                c for c in contours if cv2.contourArea(c) > min_area
            ]
            n_blocks = len(blocks)

            # Optional visualisation / logging
            if save_images:
                vis = img.copy()
                for c in blocks:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame_name = image_dir / f"ep{ep:02d}_step{step:02d}.png"
                cv2.imwrite(str(frame_name), vis)

            #print(f"  Step {step + 1:02d}: blocks={n_blocks}")

            # -------------- update state for next step ------------------
            next_irs  = rob.read_irs()
            state     = get_state(next_irs, count, grid)

    # ------------------------------------------------------------------
    # 4) Clean-up / reset pose
    # ------------------------------------------------------------------
    rob.reset_wheels()
    if sim_mode:
        rob.set_position(initial_pos, initial_ori)
        rob.stop_simulation()



def task3_test(rob):
    import time
    # ------------------------------------------------------------------
    # 2) If running in simulation, remember pose so we can reset later
    # ------------------------------------------------------------------
    sim_mode = isinstance(rob, SimulationRobobo)
    if sim_mode:
        rob.play_simulation()
        initial_pos, initial_ori = rob.get_position(), rob.get_orientation()


    rob.set_phone_tilt_blocking(109, 30)
    while True:
        # Original frame
        img = rob.read_image_front()
        flipped = cv2.flip(img, -1)
        cv2.imwrite("/root/results/red_block.png", flipped)
        # Convert Image to HSV
        hsv = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV)

        # HSV range for green
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Mask, filter, grid, contours
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        cv2.imwrite("/root/results/red_block_mask.png", mask)
        red_matrix = (mask > 0).astype(np.uint8)
        grid = downsample_mask(red_matrix, (3, 3))
        print(grid)
        time.sleep(5)
