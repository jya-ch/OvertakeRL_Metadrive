# visualize.py
import gymnasium as gym
from stable_baselines3 import PPO
from env_ot_lidar import make_gym_env
import time

# ★ 현재 학습 중인 모델 경로 (폴더명 확인해서 수정하세요!)
# 예: runs/ppo_ot_lidar_traffic/2023.../model.zip 또는 models/ppo_ot_lidar_traffic_continue.zip
MODEL_PATH = "models/checkpoints_curve/ppo_lidar_continue_11013504_steps.zip" 

def main():
    # 렌더링 켠 환경 생성
    env = make_gym_env(
        #map_sequence="SSSSSSSSSS",
        random_map= True,
        traffic_density=0.12,
        use_render=True,  # ★ 화면 켜기
        lane_num=5
    )
    
    # 모델 로드 (없으면 랜덤 행동)
    try:
        model = PPO.load(MODEL_PATH)
        print("✅ 모델 로드 성공! AI가 운전합니다.")
    except:
        model = None
        print("⚠️ 모델을 찾을 수 없습니다. 랜덤 행동으로 테스트합니다.")

    obs, info = env.reset()
    
    for i in range(10): # 10번 주행
        done = False
        print(f"=== Episode {i+1} Start ===")
        
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 진행 상황 출력
            speed = info.get('velocity_kmh', 0)
            if speed < 25:
                print(f"⚠️ 저속 경고! Speed: {speed:.1f} km/h")
            
            env.render()
            # time.sleep(0.01) # 너무 빠르면 주석 해제

        print(f"Episode Finished. Info: {info}")
        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()