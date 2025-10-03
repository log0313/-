import numpy as np
import math
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 1. 상수 및 데이터 클래스 정의
# --------------------------------------------------------------------------
@dataclass
class Constants:
    INITIAL_DISTANCE: float = 100.0
    MIN_SIGNAL_STRENGTH: float = -100.0
    MIN_DISTANCE_TO_HOTSPOT: float = 1.0
    ROTATION_ANGLE_RAD: float = np.pi / 4

@dataclass
class SimParams:
    TRANSMIT_POWER_DBM: float = 15.0
    DIST_FAR: float = 15.0
    DIST_MID: float = 5.0
    DIST_NEAR: float = 3.0
    DIST_PINPOINT: float = 1.2

    STUCK_THRESHOLD: int = 3
    ESCAPE_DISTANCE: float = 25.0

    SIGNAL_MID: float = -51.0
    SIGNAL_NEAR: float = -45.0
    SIGNAL_PINPOINT: float = -35.0
    ASCENT_THRESHOLD: float = -60.0

    PROBE_DISTANCE: float = 8.0
    GPS_DRIFT_FACTOR: float = 0.8 # 1 이면 GPS 오차 없다고 생각
    ROTATION_PENALTY_TIME: float = 1.5
    DRONE_SPEED: float = 8.0
    RSSI_SCAN_TIME: float = 2.0
    TIME_LIMIT: float = 100000.0
    GPS_ERROR_STD: float = 8.0
    RSSI_SHADOW_STD: float = 1.0
    SENSOR_DELAY_MEAN: float = 0.12
    SENSOR_DELAY_STD: float = 0.02
    SENSOR_ERROR_STD: float = 1.2
    NUM_ESCAPE_SAMPLES: int = 8
    ESCAPE_SAMPLE_RADIUS: float = 20.0

    ### 지수이동평균_평활상수 ###
    """(0~1 사이 값) 신호 필터링 강도. 높을수록 현재 값에 민감, 낮을수록 둔감."""
    RSSI_SMOOTHING_FACTOR: float = 0.4
    #########################

    # --- 페이딩 모델 파라미터 ---
    ENABLE_FADING: bool = False  
    RICIAN_K_FACTOR: float = 0  # 라이시안 K 팩터 (dB)
                                    # K > 10: LOS (페이딩 거의 없음)
                                    # K = 3~10 dB: LOS + 산란 혼합
                                    # K ≈ 0 dB: 레이리 페이딩 (NLOS)

@dataclass
class SimResult:
    """단일 시뮬레이션의 결과를 저장하는 구조체 역할"""
    success: bool
    final_distance: float
    total_travel: float
    search_time: float
    reason: str
    waypoint_count: int
    waypoints_at_threshold: int
    rssi_at_success: float

# --------------------------------------------------------------------------
# 2. Simulation Environment Class
# --------------------------------------------------------------------------
class SimulationEnvironment:
    def __init__(self, params: SimParams):
        if not isinstance(params, SimParams):
            raise TypeError("params는 SimParams의 인스턴스여야 합니다.")
        self.params = params
        angle = np.random.uniform(0, 2 * np.pi)
        self.hotspot_pos = np.array([
            Constants.INITIAL_DISTANCE * np.cos(angle),
            Constants.INITIAL_DISTANCE * np.sin(angle)
        ])

    def apply_rician_fading(self, signal_db: float) -> float:
        """
        라이시안 페이딩을 신호에 적용

        라이시안 페이딩 모델:
        수신 신호 진폭 r은 다음 확률 밀도 함수를 따름:

        f(r) = (r/σ²) * exp(-(r² + s²)/(2σ²)) * I₀(rs/σ²)

        K-factor 조절로 다양한 환경 표현:
        - K > 10 dB: 순수 LOS (페이딩 거의 없음)
        - K = 3~10 dB: LOS + 산란 혼합 환경
        - K → 0 dB: 레이리 페이딩 (순수 NLOS)
        """
        if not self.params.ENABLE_FADING:
            return signal_db

        # dB를 선형 스케일로 변환
        signal_linear = 10 ** (signal_db / 10)

        # K-factor를 선형 스케일로 변환
        K_linear = 10 ** (self.params.RICIAN_K_FACTOR / 10)

        # LOS 성분
        los_component = np.sqrt(K_linear / (K_linear + 1))

        # 산란 성분의 표준편차
        scatter_scale = 1 / np.sqrt(2 * (K_linear + 1))

        # 산란 성분 생성 
        scatter_real = np.random.normal(0, scatter_scale)  # N₁
        scatter_imag = np.random.normal(0, scatter_scale)  # N₂

        # 복소 진폭의 크기
        amplitude = np.abs(los_component + scatter_real + 1j * scatter_imag)

        # 수신 전력 = 진폭²
        faded_signal_linear = signal_linear * (amplitude ** 2)

        # 선형 스케일을 다시 dB로 변환
        if faded_signal_linear > 0:
            return 10 * np.log10(faded_signal_linear)
        else:
            return Constants.MIN_SIGNAL_STRENGTH

    def get_signal(self, pos: np.ndarray, add_noise: bool = True) -> float:
        if not isinstance(pos, np.ndarray) or pos.shape != (2,):
            raise ValueError("위치는 2D numpy 배열이어야 합니다.")

        distance = np.linalg.norm(pos - self.hotspot_pos)
        distance = max(distance, Constants.MIN_DISTANCE_TO_HOTSPOT)
        p_tx = self.params.TRANSMIT_POWER_DBM
        path_loss_db = 30.0 + 20 * np.log10(distance)
        signal = p_tx - path_loss_db

        # 라이시안 페이딩 적용
        signal = self.apply_rician_fading(signal)

        if add_noise:
            shadow_fading = np.random.normal(0, self.params.RSSI_SHADOW_STD)
            signal += shadow_fading
            small_scale_fading = np.random.randn() * 0.5
            signal += small_scale_fading

        return max(Constants.MIN_SIGNAL_STRENGTH, signal)

# --------------------------------------------------------------------------
# 3. HomingAlgorithm Class
# --------------------------------------------------------------------------
class HomingAlgorithm:
    
    def __init__(self, start_pos: np.ndarray, params: SimParams):
        self.pos, self.waypoint = np.array(start_pos, dtype=float), np.array(start_pos, dtype=float)
        self.path, self.params, self.is_finished = [start_pos.copy()], params, False
        self.state = "SPIRAL"
        self.last_signal, self.stuck_counter = Constants.MIN_SIGNAL_STRENGTH, 0
        self.best_known_pos, self.best_known_signal = self.pos.copy(), Constants.MIN_SIGNAL_STRENGTH
        self.ascent_direction = np.array([1.0, 0.0])
        self.waypoint_count: int = 0

        self.spiral_waypoint = self.pos.copy()
        self.spiral_leg_length, self.spiral_steps_taken, self.spiral_legs_completed = 1, 0, 0
        self.spiral_direction = np.array([1.0, 0.0])

        self.escape_points, self.escape_results = [], {}
        self.current_escape_point_index, self.stuck_signal_baseline = 0, Constants.MIN_SIGNAL_STRENGTH

        # Kalman Filter 초기화
        self.kf_estimate = Constants.MIN_SIGNAL_STRENGTH
        self.kf_error_cov = 1.0   # 추정 공분산
        self.process_var = 1.0    # 프로세스 잡음 (신호 변화 가정)
        self.measure_var = 4.0    # 측정 잡음 분산 (RSSI 오차 dB^2)

    def _kalman_update(self, measurement: float) -> float:
        """Kalman Filter로 RSSI 업데이트"""
        # 예측 단계
        pred_estimate = self.kf_estimate
        pred_error_cov = self.kf_error_cov + self.process_var

        # 갱신 단계
        K = pred_error_cov / (pred_error_cov + self.measure_var)   # 칼만 이득
        self.kf_estimate = pred_estimate + K * (measurement - pred_estimate)
        self.kf_error_cov = (1 - K) * pred_error_cov

        return self.kf_estimate

    def decide_action(self, rssi: float):
        if self.is_finished: 
            return
        self.waypoint_count += 1

        # Kalman Filter 적용
        filtered_rssi = self._kalman_update(rssi)

        # best-known 갱신
        if filtered_rssi > self.best_known_signal:
            self.best_known_signal, self.best_known_pos = filtered_rssi, self.pos.copy()

        if self.state == "SPIRAL":
            self._execute_spiral(filtered_rssi)
        elif self.state == "ADAPTIVE_ASCENT":
            self._execute_adaptive_ascent(filtered_rssi)
        elif self.state == "ESCAPING":
            self._execute_escaping(filtered_rssi)

        self.last_signal = filtered_rssi

    def _execute_spiral(self, rssi: float):
        if rssi > self.params.ASCENT_THRESHOLD:
            self.state = "ADAPTIVE_ASCENT"
            self._execute_adaptive_ascent(rssi)
            return
        step_distance = self.params.DIST_FAR
        self.waypoint = self.spiral_waypoint + self.spiral_direction * step_distance
        self.spiral_steps_taken += 1
        if self.spiral_steps_taken >= self.spiral_leg_length:
            self.spiral_steps_taken, self.spiral_legs_completed = 0, self.spiral_legs_completed + 1
            self.spiral_direction = np.array([-self.spiral_direction[1], self.spiral_direction[0]])
            if self.spiral_legs_completed % 2 == 0: 
                self.spiral_leg_length += 1
        self.spiral_waypoint = self.waypoint

    def _execute_adaptive_ascent(self, rssi: float):
        if rssi > self.last_signal:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1
            rot_matrix = np.array([[np.cos(Constants.ROTATION_ANGLE_RAD), -np.sin(Constants.ROTATION_ANGLE_RAD)],
                                   [np.sin(Constants.ROTATION_ANGLE_RAD), np.cos(Constants.ROTATION_ANGLE_RAD)]])
            self.ascent_direction = np.dot(rot_matrix, self.ascent_direction)

        if self.stuck_counter > self.params.STUCK_THRESHOLD:
            self._initiate_escaping(rssi)
            self.state = "ESCAPING"
            return
        self.waypoint = self.pos + self.ascent_direction * self._get_adaptive_distance(rssi)

    def _initiate_escaping(self, current_rssi: float):
        self.stuck_signal_baseline = current_rssi
        self.escape_points, self.escape_results = [], {}
        self.current_escape_point_index = 0

        num_samples, radius = self.params.NUM_ESCAPE_SAMPLES, self.params.ESCAPE_SAMPLE_RADIUS
        for _ in range(num_samples):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            point = self.best_known_pos + np.array([r * np.cos(angle), r * np.sin(angle)])
            self.escape_points.append(point)
        self.waypoint = self.escape_points[0]

    def _execute_escaping(self, rssi: float):
        if np.linalg.norm(self.pos - self.waypoint) < 2.0:
            if rssi > self.stuck_signal_baseline + 1.0:
                self.ascent_direction = (self.waypoint - self.best_known_pos)
                norm = np.linalg.norm(self.ascent_direction)
                if norm > 0: 
                    self.ascent_direction /= norm
                self.state, self.stuck_counter = "ADAPTIVE_ASCENT", 0
                return

            if self.current_escape_point_index >= len(self.escape_points) - 1:
                self.escape_results[self.current_escape_point_index] = rssi
                if self.escape_results:
                    best_point_idx = max(self.escape_results, key=self.escape_results.get)
                    self.waypoint = self.escape_points[best_point_idx]
                self.state, self.stuck_counter = "ADAPTIVE_ASCENT", 0
                return

            self.escape_results[self.current_escape_point_index] = rssi
            self.current_escape_point_index += 1
            self.waypoint = self.escape_points[self.current_escape_point_index]

    def _get_adaptive_distance(self, signal: float) -> float:
        if signal > self.params.SIGNAL_PINPOINT: return self.params.DIST_PINPOINT
        if signal > self.params.SIGNAL_NEAR: return self.params.DIST_NEAR
        if signal > self.params.SIGNAL_MID: return self.params.DIST_MID
        return self.params.DIST_FAR

    def update_position(self, new_pos: np.ndarray):
        self.pos, self.path = new_pos, self.path + [new_pos.copy()]

    def get_total_distance(self) -> float:
        return np.sum(np.linalg.norm(np.diff(np.array(self.path), axis=0), axis=1)) if len(self.path) > 1 else 0.0
# --------------------------------------------------------------------------
# 4. 시각화 클래스
# --------------------------------------------------------------------------
class SimulationVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
    def update(self, env: SimulationEnvironment, algo: HomingAlgorithm, reported_pos: np.ndarray, simulation_time: float, current_rssi: float):
        self.ax.clear()
        reported_path_arr = np.array(algo.path)
        self.ax.plot(reported_path_arr[:, 0], reported_path_arr[:, 1], 'o--', color='lightcoral', ms=2, label='Drone Path')
        self.ax.plot(reported_pos[0], reported_pos[1], 'o', color='gold', ms=10, mfc='none', mew=2, label='Reported Position')
        self.ax.plot(env.hotspot_pos[0], env.hotspot_pos[1], 'rX', markersize=12, label='Hotspot (Goal)')
        self.ax.plot(algo.waypoint[0], algo.waypoint[1], 'bo', markersize=8, mfc='none', label='Next Waypoint')
        self.ax.set_title(f"Time: {simulation_time:.1f}s | State: {algo.state} | RSSI: {current_rssi:.1f} dBm")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax.legend(loc='upper right')
        self.ax.axis('equal')
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 시각화 영역을 200m * 200m로 설정했음. 필요에 따라 넓혀도 됨.
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        plt.pause(0.1)
    def close(self):
        plt.ioff()
        plt.show()

# --------------------------------------------------------------------------
# 5. 시뮬레이션 실행 클래스
# --------------------------------------------------------------------------
class SimulationRunner:
    def __init__(self, params: SimParams):
        self.params = params

    def run_single(self, visualizer: SimulationVisualizer = None) -> SimResult:
        env = SimulationEnvironment(self.params)
        algo = HomingAlgorithm(start_pos=np.array([0.0, 0.0]), params=self.params)
        
        true_pos = np.array([0.0, 0.0])
        previous_gps_error, previous_direction = np.zeros(2), None
        waypoints_at_threshold_pass, threshold_passed = 0, False

        # --- 성공 시점의 RSSI를 기록하기 위한 변수 ---
        rssi_at_success: float = 0.0
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # --- [추가] 3-시그마 기반의 동적 성공 임계값 계산 
        # 1. 10m 지점의 이론적 RSSI 계산할 것, 핫스팟 기준이라 TX 파워를 15로 가정했음
        # 공식: 15.0 - (30 + 20 * log10(10)) = -35.0
        theoretical_rssi_at_10m = -35.0
        
        # 2. 현재 시뮬레이션의 RSSI_SHADOW_STD 값을 가져온다 (1, 3, 5 중 하나)
        current_std = self.params.RSSI_SHADOW_STD
        
        # 3. 2-시그마 값을 더해 최종 성공 임계값(상한선)을 계산
        success_threshold_dbm = theoretical_rssi_at_10m + (3 * current_std)

        # 시각화 모드일 때 계산된 임계값 출력 (확인용)
        if visualizer:
            print(f"\nSimulation with RSSI_SHADOW_STD = {current_std:.1f}")
            print(f"Success Threshold set to: {success_threshold_dbm:.2f} dBm")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        simulation_time = 0.0
        while simulation_time < self.params.TIME_LIMIT:
            new_random_error = np.random.normal(0, self.params.GPS_ERROR_STD, 2)
            drift_factor = self.params.GPS_DRIFT_FACTOR
            gps_error = drift_factor * previous_gps_error + (1 - drift_factor) * new_random_error
            previous_gps_error = gps_error

            reported_pos = true_pos + gps_error
            algo.update_position(reported_pos)

            sensor_delay = max(0.0, np.random.normal(self.params.SENSOR_DELAY_MEAN, self.params.SENSOR_DELAY_STD))
            sensor_error = np.random.normal(0, self.params.SENSOR_ERROR_STD)
            current_rssi = env.get_signal(true_pos) + sensor_error
            simulation_time += sensor_delay

            algo.decide_action(current_rssi)
            
           # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            if not algo.is_finished and current_rssi >= success_threshold_dbm:
                algo.is_finished = True
                algo.state = "FINISHED_SUCCESS"
                # 성공 시점의 RSSI 값 체크
                rssi_at_success = current_rssi
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            move_vector = algo.waypoint - reported_pos
            move_distance = np.linalg.norm(move_vector)
            move_direction = move_vector / move_distance if move_distance > 0 else None

            rotation_penalty = 0.0
            if previous_direction is not None and move_distance > 0:
                angle_change = np.arccos(np.clip(np.dot(previous_direction, move_direction), -1.0, 1.0))
                if angle_change > np.deg2rad(10):
                    rotation_penalty = self.params.ROTATION_PENALTY_TIME
            previous_direction = move_direction

            true_pos += move_vector
            time_to_travel = move_distance / self.params.DRONE_SPEED
            simulation_time += time_to_travel + self.params.RSSI_SCAN_TIME + rotation_penalty

            if visualizer:
                visualizer.update(env, algo, reported_pos, simulation_time, current_rssi)
            
            # --- [수정] 성공 판정 로직을 다시 거리 기반으로 변경 ---
            if not algo.is_finished and current_rssi >= success_threshold_dbm:
                algo.is_finished = True
                algo.state = "FINISHED_SUCCESS"
                rssi_at_success = current_rssi

            if algo.is_finished:
                break
        
        reason = "Success" if algo.is_finished else "Timeout"
        final_distance = np.linalg.norm(true_pos - env.hotspot_pos)

        return SimResult(
            success=algo.is_finished,
            final_distance=final_distance,
            total_travel=algo.get_total_distance(),
            search_time=simulation_time,
            reason=reason,
            waypoint_count=algo.waypoint_count,
            waypoints_at_threshold=waypoints_at_threshold_pass,
            # --- [수정] 결과에 성공 시점 RSSI 포함 ---
            rssi_at_success=rssi_at_success
        )

    def run_multiple(self, num_simulations: int = 1000):
        print(f"Starting {num_simulations} simulations...")
        results = []
        start_time = time.time()
        
        for i in range(num_simulations):
            if (i + 1) % 10 == 0:
                print(f"\rProgress: {i + 1}/{num_simulations}", end="")
            results.append(self.run_single())
            
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        self._analyze_results(results)

    def _analyze_results(self, results: list[SimResult]):
        successful_runs = [r for r in results if r.success]
        success_count = len(successful_runs)
        num_simulations = len(results)

        print("\n--- Final Statistical Analysis ---")
        print(f"Success Rate: {success_count / num_simulations * 100:.1f}%")
    
        if successful_runs:
            # 데이터 추출
            success_waypoint_counts = [r.waypoint_count for r in successful_runs]
            success_rssi_values = [r.rssi_at_success for r in successful_runs]
            total_travels = [r.total_travel for r in successful_runs]
            search_times = [r.search_time for r in successful_runs]
            
            # 통계 출력
            print(f"\n--- Stats on Successful Runs ({len(successful_runs)} runs) ---")
            print(f"Avg Waypoints Generated: {np.mean(success_waypoint_counts):.1f} (Std: {np.std(success_waypoint_counts):.1f})")
            print(f"Avg RSSI at Success Moment: {np.mean(success_rssi_values):.2f} dBm (Std: {np.std(success_rssi_values):.2f} dBm)")
            print(f"Avg Total Distance Traveled: {np.mean(total_travels):.2f} m (Std: {np.std(total_travels):.2f} m)")
            print(f"Avg Search Time: {np.mean(search_times):.2f} s (Std: {np.std(search_times):.2f} s)")
        
        failure_reasons = {}
        for r in results:
            if not r.success:
                failure_reasons[r.reason] = failure_reasons.get(r.reason, 0) + 1
                
        print("\n--- Failure Analysis ---")
        if not failure_reasons:
            print("No failures recorded.")
        else:
            total_failures = num_simulations - success_count
            if total_failures > 0:
                for reason, count in failure_reasons.items():
                    print(f"- {reason}: {count} times ({count / total_failures * 100:.1f}%)")

# --------------------------------------------------------------------------
# 6. Main Execution Block
# --------------------------------------------------------------------------
def main():
    mode = input(
        "Select mode:\n"
        " [1] Single Simulation (Visualized)\n"
        " [2] Multi-Simulation (Statistical Analysis for RSSI_SHADOW_STD = 1, 3, 5)\n"
        " >> "
    )

    if mode == '1':
        print("\nStarting single simulation...")
        params = SimParams()
        runner = SimulationRunner(params)
        visualizer = SimulationVisualizer()
        result = runner.run_single(visualizer=visualizer)
        visualizer.close()
        
        print("\n--- Simulation Result ---")
        print(f"Success: {result.success} (Reason: {result.reason})")
        if result.success:
            print("\n--- On Success ---")
            print(f"Waypoints Generated: {result.waypoint_count}")
            print(f"RSSI at Success Moment: {result.rssi_at_success:.2f} dBm")
            print(f"Total Distance Traveled: {result.total_travel:.2f} m")
        
        print("\n--- General Info ---")
        print(f"Final Distance to Hotspot: {result.final_distance:.2f} m")
        print(f"Total Search Time: {result.search_time:.2f} s")

    elif mode == '2':
        shadow_std_values = [1.0, 3.0, 5.0]
        for std_val in shadow_std_values:
            print("\n" + "="*50)
            print(f"### Starting Simulation for RSSI_SHADOW_STD = {std_val:.1f} ###")
            print("="*50)
            params = SimParams()
            params.RSSI_SHADOW_STD = std_val
            runner = SimulationRunner(params)
            runner.run_multiple(1000)
    
    else:
        print("Invalid input. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
