# env_ot_lidar.py

import logging
import numpy as np
import gymnasium as gym
from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

class OvertakeEnv(MetaDriveEnv):
    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        self._prev_overtake_num = 0
        self._episode_collision_count = 0
        self._episode_max_speed = 0.0

    def reset(self, *, seed=None, options=None):
        self._prev_overtake_num = 0
        self._episode_collision_count = 0
        self._episode_max_speed = 0.0
        
        if seed is not None:
            obs, info = super().reset(seed=seed)
        else:
            obs, info = super().reset()
            
        # [초기 속도 강제] 40km/h로 시작
        start_speed_kmh = 40.0
        start_speed_ms = start_speed_kmh / 3.6
        
        for _, vehicle in self.agents.items():
            heading = vehicle.heading_theta
            vx = start_speed_ms * np.cos(heading)
            vy = start_speed_ms * np.sin(heading)
            vehicle.set_velocity([vx, vy]) 
            
        return obs, info
    
    
    def done_function(self, vehicle_id: str):
        done, info = super().done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        
        current_speed_kmh = vehicle.speed * 3.6

        # [안정성] 타임아웃
        if self.episode_step >= 3000:
            done = True

        # ★ [수정] 강제 종료 기준을 20km/h -> 0.5km/h로 대폭 완화!
        # 이제 앞차 뒤에서 서행한다고 죽이지 않습니다. 아예 멈추지 않는 한 계속 기회(시간)를 줍니다.
        # 대신 reward_function에서 저속 주행 감점(-1.0)은 유지되므로, 
        # 에이전트는 "살아는 있는데 점수가 계속 까이네? 옆으로 비켜볼까?"를 고민할 시간이 생깁니다.
        if self.episode_step > 50 and current_speed_kmh < 0.5:
            done = True
            info["low_speed_termination"] = True
            
        # [종료 조건] 충돌
        if info.get("crash_vehicle", False) or info.get("crash_object", False):
            done = True
            
        return done, info

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = super().reward_function(vehicle_id)[1]
        
        speed_kmh = vehicle.speed * 3.6
        lane_width = vehicle.lane.width
        lateral_pos = vehicle.lane.local_coordinates(vehicle.position)[1]
        lane_diff_norm = abs(lateral_pos) / (lane_width / 2)
        
        # --- [보상 설계] ---
        r_speed = speed_kmh / 60.0 
        r_distance = vehicle.speed * 0.1 

        # 충돌 패널티 (범퍼카 모드: -0.2)
        r_penalty = 0.0
        is_crash = step_info.get("crash_vehicle", False) or step_info.get("crash_object", False)
        is_out = step_info.get("out_of_road", False)

        if is_crash:
            r_penalty = -0.2 
            self._episode_collision_count += 1
        
        if is_out:
            r_penalty = -1.0 

        # 정지 패널티
        r_idle_penalty = 0.0
        if speed_kmh < 1.0:
            r_idle_penalty = -0.05

        # 가속 행위 보상
        throttle = vehicle.throttle_brake
        r_throttle = 0.0
        if throttle > 0:
            r_throttle = throttle * 0.1

        total_reward = r_speed + r_distance + r_penalty + r_idle_penalty + r_throttle
        
        # --- 로깅 ---
        step_info["velocity_kmh"] = speed_kmh
        step_info["debug_speed_kmh"] = speed_kmh
        step_info["debug_collision_count"] = self._episode_collision_count
        step_info["debug_overtake_count"] = 0
        step_info["debug_is_out"] = 1.0 if is_out else 0.0
        
        step_info["reward_breakdown/speed"] = r_speed
        step_info["reward_breakdown/penalty"] = r_penalty
        
        return total_reward, step_info

def make_metadrive_env(
    map_sequence: str = "SSSSSSSSSS", 
    lane_num: int = 5,
    traffic_density: float = 0.15,
    use_render: bool = False,
    start_seed: int = 0,
    accident_prob: float = 0.0,
    random_map: bool = False,  # ★ [추가] 랜덤 맵 스위치
    **kwargs
) -> MetaDriveEnv:
    

    # ★ [수정] 랜덤 맵 로직 처리
    if random_map:
        # 랜덤 생성 모드
        # BIG_BLOCK_NUM: 블록의 개수만 정해주면 랜덤으로 조합함
        map_gen_type = MapGenerateMethod.BIG_BLOCK_NUM
        map_gen_config = 7  # 블록 7개를 랜덤으로 이어 붙임 (너무 길면 복잡하니 적당히)
    else:
        # 고정 맵 모드
        map_gen_type = MapGenerateMethod.BIG_BLOCK_SEQUENCE
        map_gen_config = map_sequence

    config = dict(
        map_config={
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: map_sequence,
            BaseMap.LANE_NUM: lane_num,
            BaseMap.LANE_WIDTH: 3.5,
        },
        traffic_density=traffic_density,

        random_traffic=True, 
        
        # ★ [추가] 장애물(사람/콘) 생성 확률 (0.0 ~ 1.0)
        # 0.0: 없음 (기존)
        # 0.5: 적당히 나옴
        # 1.0: 도로가 난장판이 됨 (난이도 급상승)
        accident_prob=accident_prob,

        vehicle_config={
            "lidar": {
                "num_lasers": 72, 
                "distance": 50, 
                "num_others": 4,
            },
            "show_lidar": False,
        },
        
        # ★ [수정됨] 문법 오류 해결 (콜론 : 대신 등호 = 사용)
        debug=False,       
        force_destroy=True,
        
        crash_vehicle_done=True,  
        crash_object_done=True,
        out_of_road_done=True,
        
        driving_reward=0.0,
        speed_reward=0.0,
        use_lateral_reward=False,
        
        use_render=use_render,
        #start_seed=start_seed,
        horizon=2000, 
    )
    env = OvertakeEnv(config)
    return env

def make_gym_env(**kwargs) -> gym.Env:
    env = make_metadrive_env(**kwargs)
    return env

if __name__ == "__main__":
    env = make_gym_env(map_sequence="S", traffic_density=1.1, use_render=False)
    try:
        obs, info = env.reset()
        print(f"[INFO] Env Test Success. Obs Shape: {obs.shape}")
        
        speed = info.get('velocity_kmh', 0.0)
        print(f"[CHECK] Start Speed: {speed:.2f} km/h")
        
        env.step([0, 1])
    except Exception as e:
        print(f"[ERROR] Env Test Failed: {e}")
    finally:
        env.close()