#mult_agent_evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from stable_baselines3 import PPO

# ★ 기존 코드의 로직을 그대로 가져오되, 상속만 MultiAgentMetaDrive로 변경
class MultiOvertakeEnv(MultiAgentMetaDrive):
    def __init__(self, config=None):
        super().__init__(config)
        self._prev_overtake_num = 0
        self._episode_collision_count = 0

    def reset(self, *, seed=None, options=None):
        # [수정됨] options=options 를 제거했습니다.
        # seed는 보통 지원하지만, 만약 seed에서도 에러가 나면 super().reset()만 남기세요.
        obs, info = super().reset(seed=seed)
        
        # [초기 속도 강제 설정]
        start_speed_kmh = 40.0
        start_speed_ms = start_speed_kmh / 3.6
        
        if self.agents:
            for v_id, vehicle in self.agents.items():
                # 차량 객체가 생성된 직후이므로 물리 엔진 적용을 위해 속도 설정
                heading = vehicle.heading_theta
                vx = start_speed_ms * np.cos(heading)
                vy = start_speed_ms * np.sin(heading)
                vehicle.set_velocity([vx, vy])
                
        return obs, info

    def done_function(self, vehicle_id: str):
        # 작성하신 종료 조건 그대로 사용
        done, info = super().done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        current_speed_kmh = vehicle.speed * 3.6

        # 타임아웃
        if self.episode_step >= 3000:
            done = True
        
        # 저속 종료
        if self.episode_step > 50 and current_speed_kmh < 0.5:
            done = True
            info["low_speed_termination"] = True
            
        if info.get("crash_vehicle", False) or info.get("crash_object", False):
            done = True
            
        return done, info

    def reward_function(self, vehicle_id: str):
        # 작성하신 보상 함수 그대로 사용 (멀티 에이전트에서도 vehicle_id로 구분되어 작동함)
        vehicle = self.agents[vehicle_id]
        step_info = super().reward_function(vehicle_id)[1]
        
        speed_kmh = vehicle.speed * 3.6
        
        # 보상 계산 로직 (작성하신 코드 복사)
        r_speed = speed_kmh / 60.0 
        r_distance = vehicle.speed * 0.1 
        
        r_penalty = 0.0
        is_crash = step_info.get("crash_vehicle", False) or step_info.get("crash_object", False)
        is_out = step_info.get("out_of_road", False)

        if is_crash: r_penalty = -0.2 
        if is_out: r_penalty = -1.0 

        r_idle_penalty = 0.0
        if speed_kmh < 1.0: r_idle_penalty = -0.05

        throttle = vehicle.throttle_brake
        r_throttle = 0.0
        if throttle > 0: r_throttle = throttle * 0.1

        total_reward = r_speed + r_distance + r_penalty + r_idle_penalty + r_throttle
        
        step_info["velocity_kmh"] = speed_kmh
        return total_reward, step_info

# ==========================================
# 실행부 (Main)
# ==========================================

# 모델 경로 (본인의 경로로 수정)
MODEL_PATH = "models/ppo_ot_lidar_traffic_final.zip"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading Model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # 1. 환경 설정
    env_config = {
        "use_render": True,
        "map_config": {
            "type": "block_sequence",
            "config": "SSSSSSSSS", 
            "lane_num": 5,
            "lane_width": 3.5
        },
        "traffic_density": 0.06,  
        "num_agents": 3,         
        
        # ★ [핵심 수정] 위치 및 앞뒤 간격 조정
        "agent_configs": {
            "agent0": {
                # 2번째 차선(인덱스 1), 출발선(0m) 위치
                "spawn_lane_index": (">>", ">>>", 3), 
                "spawn_longitude": 0   
            },
            "agent1": {
                # 3번째 차선(인덱스 2), 출발선보다 15m 앞쪽 위치
                "spawn_lane_index": (">>", ">>>", 1), 
                "spawn_longitude": 50 
            },
            "agent3": {
                # 3번째 차선(인덱스 2), 출발선보다 15m 앞쪽 위치
                "spawn_lane_index": (">>", ">>>", 5), 
                "spawn_longitude": 50 
            },
        },
        
        "accident_prob": 0.0,
        "debug": False,
        "crash_done": True, # 충돌 시 해당 에이전트 종료
        
        "vehicle_config": {
            "lidar": {
                "num_lasers": 72, 
                "distance": 50, 
                "num_others": 4,
            },
            "show_lidar": False,
        }
    }

    env = MultiOvertakeEnv(env_config)

    NUM_EPISODES = 5
    
    try:
        for ep in range(NUM_EPISODES):
            # reset() 호출 시 options 제거됨 (클래스 수정 완료 가정)
            obs, info = env.reset()
            
            print(f"\nEpisode {ep+1} Start! Agents: {list(obs.keys())}")
            
            done = False
            step = 0
            
            while not done:
                actions = {}
                
                # 살아있는 에이전트들만 액션 결정
                for agent_id, agent_obs in obs.items():
                    action, _ = model.predict(agent_obs, deterministic=True)
                    actions[agent_id] = action
                
                # 환경 Step
                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1
                
                # 렌더링
                env.render(mode="top_down") 
                
                # ★ [수정됨] 종료 조건 체크
                # terminated["__all__"] 또는 truncated["__all__"]이 True면 에피소드 종료
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    print(f"Episode {ep+1} Finished at step {step}")
                    done = True

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc() # 에러 발생 시 자세한 원인 출력
        
    finally:
        env.close()

if __name__ == "__main__":
    main()