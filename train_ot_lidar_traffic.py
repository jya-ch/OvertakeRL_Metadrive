# train_ot_lidar_traffic.py

import os
import gymnasium as gym
from datetime import datetime
import numpy as np
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from env_ot_lidar import make_gym_env

# ===========================================================
# 1. 설정 (RTX 4090 / 5000 Ada 고성능 최적화)
# ===========================================================
N_ENVS = 16  
TOTAL_TIMESTEPS = 5_000_000 
SEED = 0

# ★ [핵심] 고사양 GPU를 위한 "빅 배치" 전략
# 데이터를 아주 많이 모아서(Rollout), 한 번에 GPU로 계산(Train)합니다.
# Buffer Size = 16 * 2048 = 32,768 (메모리 충분함)
N_STEP = 1024       
BATCH_SIZE = 512   # GPU가 한 입에 씹어먹기 좋은 크기

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("runs", "ppo_ot_lidar_traffic", RUN_ID)
CONTINUED_MODEL_PATH = "models/ppo_ot_lidar_traffic_final.zip"

# ===========================================================
# 2. 콜백
# ===========================================================
class NoSeedResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return self.env.reset(**kwargs)

class DetailedLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.temp_stats = defaultdict(list)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "debug_speed_kmh" in info: self.temp_stats["speed_kmh"].append(info["debug_speed_kmh"])
            if "debug_collision_count" in info: self.temp_stats["collision_count"].append(info["debug_collision_count"])
            if "debug_overtake_count" in info: self.temp_stats["overtake_count"].append(info["debug_overtake_count"])
            
            if "reward_breakdown/speed" in info: self.temp_stats["rew_speed"].append(info["reward_breakdown/speed"])
            if "reward_breakdown/penalty" in info: self.temp_stats["rew_penalty"].append(info["reward_breakdown/penalty"])
            if "reward_breakdown/overtake" in info: self.temp_stats["rew_overtake"].append(info["reward_breakdown/overtake"])

        # 출력 빈도 조절 (5000 step 마다)
        if self.n_calls % 5000 == 0:
            avg_spd = np.mean(self.temp_stats["speed_kmh"][-100:]) if self.temp_stats["speed_kmh"] else 0
            max_col = np.max(self.temp_stats["collision_count"][-100:]) if self.temp_stats["collision_count"] else 0
            print(f" >> [Step {self.n_calls}] AvgSpeed: {avg_spd:.1f} km/h | MaxCol: {max_col}")
        return True

    def _on_rollout_end(self):
        if self.temp_stats["speed_kmh"]:
            self.logger.record("custom/status_speed_avg", np.mean(self.temp_stats["speed_kmh"]))
        if self.temp_stats["collision_count"]:
            self.logger.record("custom/status_collision_mean", np.mean(self.temp_stats["collision_count"]))
        if self.temp_stats["overtake_count"]:
            self.logger.record("custom/status_overtake_max", np.max(self.temp_stats["overtake_count"]))
        if self.temp_stats["rew_speed"]:
            self.logger.record("custom/reward_speed", np.mean(self.temp_stats["rew_speed"]))
        
        self.temp_stats = defaultdict(list)

def make_env_fn(rank: int):
    def _init():
        env = make_gym_env(
            map_sequence="SSSSSSSSSS",
            traffic_density=0.15, 
            use_render=False,
            lane_num=5,
            start_seed=rank * 1000 
        )
        env = NoSeedResetWrapper(env)
        env = Monitor(env)
        env.reset()
        return env
    return _init

# ===========================================================
# 3. 메인 학습 루프
# ===========================================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)

    env_fns = [make_env_fn(i) for i in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns)

    print(f"[Train] HIGH-PERFORMANCE MODE: CUDA + Large Batch")
    print(f"[Train] N_ENVS={N_ENVS}, N_STEP={N_STEP}, BATCH={BATCH_SIZE}")

    # [정책 파라미터] 탐험 강화 + 뇌 용량 증가
    policy_kwargs = dict(
        log_std_init=0.0, 
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4, 
        n_steps=N_STEP,       # 2048 (길게 수집)
        batch_size=BATCH_SIZE, # 2048 (한방에 학습)
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        ent_coef=0.05,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        device="cpu"         # ★★★ 4090/5000Ada는 무조건 CUDA ★★★
    )

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=os.path.join("models", "checkpoints"),
        name_prefix="ppo_lidar_fast"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=True,
        progress_bar=True,
        tb_log_name="PPO_FAST_RUN",
        callback=[DetailedLogCallback(), checkpoint_callback],
    )

    print("[Train] Saving final model:", CONTINUED_MODEL_PATH)
    model.save(CONTINUED_MODEL_PATH)
    train_env.close()

if __name__ == "__main__":
    main()