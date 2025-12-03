# human_ot_manual.py
import numpy as np
from env_ot_lidar import make_metadrive_env

if __name__ == "__main__":
    # 렌더 켜고, 우리 커스텀 env 그대로 사용
    env = make_metadrive_env(
        map_sequence="SSSSSSSSSS",
        traffic_density=0.0,
        use_render=True,
    )

    # ★ 중요: reset 전에 manual_control 켜기
    env.config["manual_control"] = True

    obs, info = env.reset()
    done = False
    step = 0
    ep_reward = 0.0

    print("Manual control mode 시작.")
    print("보통 W/S: 가속/브레이크, A/D: 조향 (또는 방향키). ESC 로 창 닫기.\n")

    while not done:
        # action 값은 무시되고 키보드 입력이 사용됨 (manual_control=True 일 때)
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step += 1
        ep_reward += float(reward)

        speed = info.get("velocity_kmh", 0.0)
        lane_offset = info.get("lane_offset", 0.0)
        heading = info.get("heading_diff", 0.0)
        offroad = info.get("offroad_flag", 0.0)

        print(
            f"step={step:4d}  "
            f"r={reward:7.3f}  "
            f"speed={speed:6.2f} km/h  "
            f"lane={lane_offset:+5.2f}  "
            f"heading={heading:+5.2f}  "
            f"offroad={offroad:.1f}"
        )

    print(f"[Manual Test] len={step}, total_reward={ep_reward:.2f}")
    env.close()
