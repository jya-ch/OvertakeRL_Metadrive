#Overtake Driving RL_metadrive

심층 강화학습(Deep Reinforcement Learning, PPO)을 활용하여 MetaDrive 시뮬레이터 환경에서 고속 추월 및 충돌 회피를 수행하는 자율주행 에이전트 프로젝트입니다.
한양대학교 강화학습 이론과 응용 수업 프로젝트 결과로 제작되었고
metadrive로 구현되어 있습니다.  (https://metadriverse.github.io/metadrive/)

---

## 주요기능
이 프로젝트는 트래픽 환경에서 에이전트가 스스로 판단하여  주변 차량을 추월하고 빠르게 주행하는 것을 목표로 합니다.
- 고속 추월 (High-speed Overtaking): 느린 차량을 인식하고 빈 차선으로 변경하여 추월
- 충돌 회피 (Collision Avoidance): Lidar 센서를 활용한 동적 장애물 감지 및 회피
- 안정적 주행 (Stable Control): 최고 속도 60km/h 상황에서의 차선 유지 및 제어

---
## Tech Stack (기술 스택)
- Environment:** [MetaDrive](https://github.com/metadriverse/metadrive) v0.4.3
- Algorithm:** PPO (Proximal Policy Optimization)
- Library:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), Gymnasium, PyTorch
- Python 3.9

---
## 실행 방법

Config Environment (환경 설정)
- env_ot_lidar.py 보상함수 및 환경 설정.

Training (학습 시작)
- 처음부터 학습을 시작합니다.
python train_ot_lidar_traffic.py

Continue Training (이어하기)
- 저장된 모델을 불러와 커리큘럼 학습(Curriculum Learning)을 진행합니다.
python continue_train.py

Evaluation / Visualization (시각화)
- 학습된 에이전트가 실제로 주행하는 모습을 렌더링 화면으로 확인합니다.
python evaluate_ot_lidar.py

File Structure
- train_ot_lidar_traffic.py: 메인 학습 스크립트 
- env_ot_lidar.py: 커스텀 MetaDrive 환경 설정 (Lidar 72ch, Reward Function)
- continue_train.py: 학습 모델 로드 및 추가 학습 스크립트
- evaluate_ot_lidar.py: 주행 시각화 및 평가 스크립트
- models/: 학습된 모델 체크포인트 저장 폴더
