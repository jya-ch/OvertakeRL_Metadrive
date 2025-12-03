# continue_train.py

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

# ★ 사용자가 제공한 env_ot_lidar.py (Lidar 72개 버전)를 가져옵니다.
from env_ot_lidar import make_gym_env

# ===========================================================
# 1. 설정 (기존 train 코드와 N_ENVS 일치 필수)
# ===========================================================
N_ENVS = 16  # 기존 학습과 동일하게 16개 유지
ADDITIONAL_TIMESTEPS = 7_000_000 # 추가로 500만 스텝 더 (총 1000만)
SEED = 0

# 경로 설정
LOAD_MODEL_PATH = "models/ppo_ot_lidar_traffic_final.zip" # 기존에 학습 완료된 모델
NEW_MODEL_PATH = "models/ppo_ot_lidar_traffic_10m_curve.zip"    # 새로 저장될 1000만 스텝 모델

# 로그 폴더 (새로운 폴더에 저장되지만, 텐서보드는 시간축을 이어줍니다)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_continue")
LOG_DIR = os.path.join("runs", "ppo_ot_lidar_traffic", RUN_ID)

# ===========================================================
# 2. 콜백 (기존 코드와 동일하게 유지)
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

        # 출력 빈도 5000
        if self.n_calls % 5000 == 0:
            avg_spd = np.mean(self.temp_stats["speed_kmh"][-100:]) if self.temp_stats["speed_kmh"] else 0
            max_col = np.max(self.temp_stats["collision_count"][-100:]) if self.temp_stats["collision_count"] else 0
            # [Continue] 태그로 구분
            print(f" >> [Continue Step {self.n_calls}] AvgSpeed: {avg_spd:.1f} km/h | MaxCol: {max_col}")
        return True

    def _on_rollout_end(self):
        if self.temp_stats["speed_kmh"]: self.logger.record("custom/status_speed_avg", np.mean(self.temp_stats["speed_kmh"]))
        if self.temp_stats["collision_count"]: self.logger.record("custom/status_collision_mean", np.mean(self.temp_stats["collision_count"]))
        if self.temp_stats["overtake_count"]: self.logger.record("custom/status_overtake_max", np.max(self.temp_stats["overtake_count"]))
        if self.temp_stats["rew_speed"]: self.logger.record("custom/reward_speed", np.mean(self.temp_stats["rew_speed"]))
        self.temp_stats = defaultdict(list)

def make_env_fn(rank: int):
    def _init():
        # 기존 학습 환경과 동일 설정 (traffic 0.15, lidar 72)
        env = make_gym_env(
            #map_sequence="SSSSSSSSSS",
            random_map=True,
            traffic_density=0.12, 
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
# 3. 메인 (Load & Continue)
# ===========================================================
def main():
    if not os.path.exists(LOAD_MODEL_PATH):
        print(f"[Error] 모델 파일이 없습니다: {LOAD_MODEL_PATH}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)

    # 1. 환경 생성
    env_fns = [make_env_fn(i) for i in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns)

    print(f"[Train] Loading Model from {LOAD_MODEL_PATH}...")
    
    # 2. 모델 불러오기
    # device="cpu" 유지 (기존 코드가 cpu였으므로)
    model = PPO.load(
        LOAD_MODEL_PATH, 
        env=train_env, 
        device="cpu", 
        print_system_info=True
    )

    # 3. 로거 연결 (새로운 로그 폴더)
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    print(f"[Train] Continuing training for {ADDITIONAL_TIMESTEPS} steps...")

    # 4. 체크포인트 콜백 설정 (100만 스텝마다 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=os.path.join("models", "checkpoints_curve"),
        name_prefix="ppo_lidar_continue"
    )

    # 5. 학습 재개
    # reset_num_timesteps=False : 0부터가 아니라 기존 스텝에 이어서 카운트
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False, 
        progress_bar=True,
        tb_log_name="PPO_CONTINUE_RUN",
        callback=[DetailedLogCallback(), checkpoint_callback],
    )

    print("[Train] Saving 10M model:", NEW_MODEL_PATH)
    model.save(NEW_MODEL_PATH)
    train_env.close()

if __name__ == "__main__":
    main()