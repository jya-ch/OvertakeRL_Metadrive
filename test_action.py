# test_action.py
import numpy as np
from env_ot_lidar import make_gym_env

def test_full_throttle():
    # 1. 환경 생성
    env = make_gym_env(
        map_sequence="SSSSSSSSSS",
        traffic_density=0.1,
        use_render=False, # 렌더링 끄고 로그로만 확인
        lane_num=5,
    )
    
    obs, info = env.reset()
    print("=== [TEST] 액션 테스트 시작 (강제 풀악셀) ===")
    
    for i in range(100):
        # 2. 강제 액션: [조향=0.0(직진), 가속=1.0(풀악셀)]
        action = np.array([0.0, 1.0], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 속도 확인
        speed = info.get("velocity_kmh", 0.0)
        print(f"Step {i+1}: Speed {speed:.2f} km/h")
        
        if speed > 20.0:
            print("\n✅ [성공] 차가 정상적으로 가속됩니다! (물리 엔진 정상)")
            env.close()
            return

    print("\n❌ [실패] 100스텝 동안 풀악셀을 밟았는데 속도가 안 납니다.")
    env.close()

if __name__ == "__main__":
    test_full_throttle()