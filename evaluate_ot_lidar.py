# evaluate_ot_lidar.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from gymnasium import Wrapper
from stable_baselines3 import PPO
from env_ot_lidar import make_gym_env

# ★ 체크포인트 경로 확인 필수 (없으면 최신 체크포인트로 변경하세요)
MODEL_PATH = "models/ppo_ot_lidar_traffic_final.zip" 
#MODEL_PATH = "models/ppo_ot_lidar_traffic_10m.zip" 
#MODEL_PATH = "models/checkpoints_curve/ppo_lidar_continue_11013504_steps.zip" 

class NoSeedResetWrapper(Wrapper):
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return self.env.reset(**kwargs)

def plot_episode(ep_idx, speeds, rewards, collisions, overtakes):
    steps = np.arange(1, len(speeds) + 1)
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Evaluation - Episode {ep_idx}", fontsize=14)

    plt.subplot(3, 1, 1)
    plt.plot(steps, speeds, color='blue', label='Speed')
    plt.ylabel("Speed [km/h]")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(steps, rewards, color='green', alpha=0.6)
    plt.ylabel("Step Reward")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(steps, collisions, color='red', label='Collisions')
    plt.plot(steps, overtakes, color='purple', label='Overtakes', linestyle='--')
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[Eval] Error: Model not found at {MODEL_PATH}")
        return

    print(f"[Eval] Loading: {MODEL_PATH}")
    # GPU 모델이지만 CPU에서 돌리려면 device="cpu" 명시 가능
    model = PPO.load(MODEL_PATH, print_system_info=True)

    env = make_gym_env(
        map_sequence="SSSSSSSSS",
        traffic_density=0.12, # 학습 환경과 동일하게
        #accident_prob = 0.8,
        use_render=True,      # 렌더링 켬
        lane_num=5
    )
    env = NoSeedResetWrapper(env)

    NUM_EPISODES = 5

    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n[Eval] Episode {ep}/{NUM_EPISODES} Start!")
        obs, info = env.reset()
        done = False
        ep_len = 0
        ep_total_reward = 0.0
        
        log_speeds, log_rewards, log_collisions, log_overtakes = [], [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action_to_env = action.tolist()
            else:
                action_to_env = action

            obs, reward, terminated, truncated, info = env.step(action_to_env)
            done = terminated or truncated
            ep_len += 1
            ep_total_reward += float(reward)

            speed = info.get("velocity_kmh", 0.0)
            col_count = info.get("debug_collision_count", 0)
            ovt_count = info.get("debug_overtake_count", 0)

            log_speeds.append(speed)
            log_rewards.append(float(reward))
            log_collisions.append(col_count)
            log_overtakes.append(ovt_count)

            if ep_len % 20 == 0:
                print(f"\rStep={ep_len:4d} | Speed={speed:5.1f}km/h | Rw={reward:5.2f} | Col={col_count} | Ovt={ovt_count}", end="")

        print(f"\n[Eval] Episode {ep} Finished. Total Reward: {ep_total_reward:.2f}")
        plot_episode(ep, log_speeds, log_rewards, log_collisions, log_overtakes)

    env.close()

if __name__ == "__main__":
    main()