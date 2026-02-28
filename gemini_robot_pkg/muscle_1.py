#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
muscle_workflow_abs_pose.py
- muscle_autoretry_reset.py 기반 + "절대좌표 up→down→회전" 시퀀스 반영 + wait/driver action 추가

요구사항 반영
1) timer/relay 지정 라벨이 들어오면
   - 먼저 label의 up pose(절대좌표)로 이동
   - 다음 label pose(절대좌표)로 이동
   - screw_motion(회전 + 케이블 역회전) 수행
   - (return_screwdriver는 brain task에서 별도 호출)

2) 그 외 라벨(unscrew/screw로 들어왔는데 절대좌표 매핑이 없으면)
   - 안전을 위해 P_SCAN으로 이동 후 20초 대기 (요구사항)

3) wait action 지원
   - action=="wait" 인 경우 task["wait_sec"] 만큼 P_SCAN에서 대기

4) driver 전달 지원
   - action=="pick_driver": nerve가 계산한 robot_x/y/z 기반으로 픽업
   - action=="handover_driver_home": P_SCAN으로 가서 그리퍼 오픈

기타
- 기존 QoS 호환 구독(TRANSIENT_LOCAL + VOLATILE), job_done publish 유지
- "노드 껐다 켜야 다음 단계 진행" 문제를 소프트리셋/재시도로 대체
"""

import os
import time
import json
import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import DR_init
from .onrobot import RG


# ==========================================
# 하드웨어 및 공정 설정 상수
# ==========================================
ROBOT_ID = os.getenv("ROBOT_ID", "dsr01").strip() or "dsr01"
ROBOT_MODEL = os.getenv("ROBOT_MODEL", "m0609").strip() or "m0609"

ROBOT_TOOL = os.getenv("ROBOT_TOOL", "Tool Weight")
ROBOT_TCP = os.getenv("ROBOT_TCP", "GripperDA_v1")

VELOCITY = int(os.getenv("MUSCLE_VELOCITY", "100"))
ACC = int(os.getenv("MUSCLE_ACC", "80"))

GRIPPER_NAME = os.getenv("GRIPPER_NAME", "rg2")
TOOLCHARGER_IP = os.getenv("TOOLCHARGER_IP", "192.168.1.1")
TOOLCHARGER_PORT = os.getenv("TOOLCHARGER_PORT", "502")

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

# 스캔/대기 포즈 (홈)
P_SCAN = [423.92, -147.13, 402.39, 168.05, -179.78, 167.78]

# 드라이버 픽업용 자세(기본은 P_SCAN 자세 방향)
DRIVER_RX = float(os.getenv("MUSCLE_DRIVER_RX", str(P_SCAN[3])))
DRIVER_RY = float(os.getenv("MUSCLE_DRIVER_RY", str(P_SCAN[4])))
DRIVER_RZ = float(os.getenv("MUSCLE_DRIVER_RZ", str(P_SCAN[5])))

DRIVER_APPROACH_DZ = float(os.getenv("MUSCLE_DRIVER_APPROACH_DZ", "120.0"))  # mm
DRIVER_GRASP_DZ = float(os.getenv("MUSCLE_DRIVER_GRASP_DZ", "0.0"))          # mm (robot_z 기준)
DRIVER_GRIP_WIDTH = int(os.getenv("MUSCLE_DRIVER_GRIP_WIDTH", "26"))         # RG2 close width
DRIVER_OPEN_WIDTH = int(os.getenv("MUSCLE_DRIVER_OPEN_WIDTH", "70"))

# 기본 대기(unknown label screw) 시간
UNKNOWN_WAIT_SEC = float(os.getenv("MUSCLE_UNKNOWN_WAIT_SEC", "20.0"))


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and x is not None


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _service_names(node: Node):
    try:
        return [name for name, _types in node.get_service_names_and_types()]
    except Exception:
        return []


def _has_service(node: Node, keyword: str) -> bool:
    for name in _service_names(node):
        if keyword in name:
            return True
    return False


def _has_robot_namespace_service(node: Node, robot_id: str) -> bool:
    rid = f"/{robot_id}/"
    for name in _service_names(node):
        if rid in name:
            return True
    return False


# ==========================================
# 절대좌표 맵(요구사항)
# ==========================================
# NOTE: 사용자 입력의 "397,28" / "158,59" 같은 값은 397.28 / 158.59로 해석
ABS_POSE_MAP: Dict[str, Tuple[List[float], List[float]]] = {
    # timer
    "timer 6": ([397.28, -91.27, 230.88, 169.39, 179.75, 168.73],
                [397.28, -91.27, 180.88, 169.39, 179.75, 168.73]),
    "timer 7": ([399.11, -144.44, 231.72, 160.22, 180.00, 160.04],
                [399.11, -144.44, 181.72, 160.22, 180.00, 160.04]),
    "timer 8": ([408.12, -144.50, 231.52, 175.10, 180.00, 174.85],
                [408.12, -144.50, 181.52, 175.10, 180.00, 174.85]),
    "timer 2": ([436.06, -143.38, 232.42, 167.30, 179.95, 166.81],
                [436.06, -143.38, 182.42, 167.30, 179.95, 166.81]),
    # relay
    "relay 4": ([220.61, -92.01, 228.74, 159.12, 180.00, 158.59],
                [220.61, -92.01, 178.74, 159.12, 180.00, 158.59]),
    "relay 3": ([232.20, -92.89, 226.86, 156.66, 180.00, 155.75],
                [232.20, -92.89, 178.86, 156.66, 180.00, 155.75]),
    "relay 5": ([233.96, -149.49, 227.44, 147.03, 180.00, 146.33],
                [233.96, -149.49, 177.44, 147.03, 180.00, 146.33]),
    "relay 6": ([222.70, -149.15, 226.98, 122.94, -179.94, 122.47],
                [222.70, -149.15, 176.98, 122.94, -179.94, 122.47]),
    "relay 7": ([223.80, -139.31, 237.60, 148.37, 180.00, 147.43],
                [223.80, -139.31, 187.60, 148.37, 180.00, 147.43]),
    "relay 8": ([222.08, -139.66, 237.18, 142.70, -180.00, 142.16],
                [222.08, -139.66, 187.18, 142.70, -180.00, 142.16]),
}


class MuscleNode(Node):
    def __init__(self):
        node_ns = os.getenv("MUSCLE_NODE_NAMESPACE", ROBOT_ID).strip().strip("/")
        super().__init__('muscle_node', namespace=node_ns)

        # QoS (라치): 늦게 떠도 마지막 작업 수신
        self.qos_latched = QoSProfile(depth=1)
        self.qos_latched.history = HistoryPolicy.KEEP_LAST
        self.qos_latched.reliability = ReliabilityPolicy.RELIABLE
        self.qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.cb_group = ReentrantCallbackGroup()

        # Job 큐/캐시
        self._lock = threading.Lock()
        self._jobs: Dict[int, List[dict]] = {}
        self._queue: Deque[int] = deque()
        self._executing_job_id: Optional[int] = None
        self.last_executed_job_id = -1

        # ✅ 재시도 카운트
        self._retry_count: Dict[int, int] = {}
        self.max_retry = int(os.getenv("MUSCLE_MAX_RETRY", "2"))          # 총 시도 횟수(기본 2)
        self.retry_delay = float(os.getenv("MUSCLE_RETRY_DELAY_SEC", "0.5"))
        self.reset_after_job = _to_bool(os.getenv("MUSCLE_RESET_AFTER_JOB", "1"))
        self.reset_on_fail = _to_bool(os.getenv("MUSCLE_RESET_ON_FAIL", "1"))

        self.paused = False

        # Doosan 준비/초기화 상태
        self._dsr_inited = False
        self._dsr_ready_timeout = float(os.getenv("MUSCLE_DSR_READY_TIMEOUT_SEC", "30.0"))
        self._call_set_robot_mode = _to_bool(os.getenv("MUSCLE_CALL_SET_ROBOT_MODE", "1"))

        # 디버그
        self.get_logger().info(
            f"🌐 ROS_DOMAIN_ID={os.getenv('ROS_DOMAIN_ID','(unset)')}, "
            f"RMW_IMPLEMENTATION={os.getenv('RMW_IMPLEMENTATION','(unset)')}"
        )
        self.get_logger().info(f"🤖 ROBOT_ID={ROBOT_ID}, ROBOT_MODEL={ROBOT_MODEL}, node_ns=/{node_ns or ''}")
        self.get_logger().info(
            f"🔁 retry: max_retry={self.max_retry}, retry_delay={self.retry_delay}s, "
            f"reset_after_job={self.reset_after_job}, reset_on_fail={self.reset_on_fail}"
        )

        # ROS 통신
        self.sub = self.create_subscription(
            String, '/nerve/robot_coords', self.coord_callback, self.qos_latched,
            callback_group=self.cb_group
        )
        # QoS 호환(퍼블리셔가 VOLATILE일 때도 수신)
        self.sub_compat = self.create_subscription(
            String, '/nerve/robot_coords', self.coord_callback, 10,
            callback_group=self.cb_group
        )

        self.pose_pub = self.create_publisher(String, "/muscle/current_pose", 10)

        self.sub_pause = self.create_subscription(
            String, '/ear/speech_text', self.pause_callback, 10,
            callback_group=self.cb_group
        )

        self.job_done_pub = self.create_publisher(String, "/muscle/job_done", 10)

        self.get_logger().info("📡 [Muscle] ready. Waiting /nerve/robot_coords")

    # =========================
    # ROS 콜백
    # =========================
    def coord_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            tasks = data if isinstance(data, list) else [data]
            if not tasks:
                return

            job_id = _safe_int(tasks[0].get("job_id", 0), default=0)

            with self._lock:
                self._jobs[job_id] = tasks
                if job_id != self.last_executed_job_id and job_id != self._executing_job_id and job_id not in self._queue:
                    self._queue.append(job_id)

            self.get_logger().info(f"📥 Job 수신/갱신: job_id={job_id}, steps={len(tasks)}")

        except Exception as e:
            self.get_logger().error(f"❌ /nerve/robot_coords 파싱 에러: {e}")

    def pause_callback(self, msg: String):
        text = (msg.data or "").lower()
        if any(word in text for word in ["멈춰", "정지", "stop"]):
            self.paused = True
            try:
                from DSR_ROBOT2 import stop, STOP_TYPE_QUICK
                stop(STOP_TYPE_QUICK)
            except Exception as e:
                self.get_logger().warn(f"stop 호출 실패: {e}")
            self.get_logger().warn("🛑 음성 정지 명령 수신!")
        elif any(word in text for word in ["계속", "재개", "resume"]):
            self.paused = False
            self.get_logger().info("▶️ 작업 재개")

    def trigger_pose_update(self):
        try:
            from DSR_ROBOT2 import get_current_posx
            pos = get_current_posx()[0]
            msg = f"{pos[0]},{pos[1]},{pos[2]},{pos[3]},{pos[4]},{pos[5]}"
            self.pose_pub.publish(String(data=msg))
        except Exception as e:
            self.get_logger().error(f"Pose 발행 실패: {e}")

    # =========================
    # Doosan 초기화/리셋
    # =========================
    def _wait_dsr_graph_ready(self) -> bool:
        end = time.time() + max(0.1, self._dsr_ready_timeout)
        while rclpy.ok() and time.time() < end:
            if _has_robot_namespace_service(self, ROBOT_ID):
                return True
            time.sleep(0.2)
        return _has_robot_namespace_service(self, ROBOT_ID)

    def _try_init_dsr(self) -> bool:
        if self._dsr_inited:
            return True

        if not self._wait_dsr_graph_ready():
            self.get_logger().warn(
                "⚠️ Doosan 서비스가 ROS 그래프에서 보이지 않습니다. "
                "dsr_bringup2_rviz.launch 실행/ROBOT_ID/ROS_DOMAIN_ID/네트워크를 확인하세요."
            )
            return False

        try:
            from DSR_ROBOT2 import (
                posx, movel, wait,
                set_tool, set_tcp,
                set_robot_mode, ROBOT_MODE_AUTONOMOUS, ROBOT_MODE_MANUAL
            )

            if self._call_set_robot_mode and _has_service(self, "set_robot_mode"):
                set_robot_mode(ROBOT_MODE_MANUAL); wait(0.5)
                set_robot_mode(ROBOT_MODE_AUTONOMOUS); wait(1.0)
            else:
                self.get_logger().info("ℹ️ set_robot_mode 스킵")

            if _has_service(self, "set_tool"):
                set_tool(ROBOT_TOOL)
            if _has_service(self, "set_tcp"):
                set_tcp(ROBOT_TCP)

            movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
            self.get_logger().info("🏠 로봇 대기 모드 진입")

            self._dsr_inited = True
            return True
        except Exception as e:
            self.get_logger().error(f"❌ DSR 초기화 실패: {e}")
            self._dsr_inited = False
            return False

    def _soft_reset(self, reason: str):
        """노드 재시작 대신: 로봇을 안전 위치로 보내고 다음 init을 강제."""
        self.get_logger().warn(f"♻️ soft reset: {reason}")

        # 그리퍼를 가능한 한 안전하게 열어둠
        try:
            gripper.set_target_width(DRIVER_OPEN_WIDTH)
            gripper.open_gripper()
        except Exception:
            pass

        # 로봇을 홈으로
        try:
            from DSR_ROBOT2 import posx, movel
            movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
        except Exception:
            pass

        self._dsr_inited = False

    # =========================
    # 모션들
    # =========================
    def pickup_screwdriver(self):
        from DSR_ROBOT2 import posx, movel, wait

        # 사용자가 제공한 return 위치와 동일 좌표계 사용
        P_TOOL_UP = posx(424.62, 224.29, 352.75, P_SCAN[3], P_SCAN[4], P_SCAN[5])
        P_TOOL_PICK = posx(424.62, 224.29, 88.75, P_SCAN[3], P_SCAN[4], P_SCAN[5])

        gripper.set_target_width(DRIVER_OPEN_WIDTH)
        gripper.open_gripper()
        wait(1.0)

        movel(P_TOOL_UP, vel=VELOCITY, acc=ACC)
        movel(P_TOOL_PICK, vel=VELOCITY, acc=ACC)

        gripper.set_target_width(DRIVER_GRIP_WIDTH)
        gripper.close_gripper()
        wait(1.5)

        movel(P_TOOL_UP, vel=VELOCITY, acc=ACC)
        self.get_logger().info("🪛 드라이버 픽업 완료")

    def return_screwdriver(self):
        from DSR_ROBOT2 import posx, movel, wait

        # 사용자 제공 코드 반영
        P_TOOL_UP = posx(424.62, 224.29, 352.75, P_SCAN[3], P_SCAN[4], P_SCAN[5])
        P_TOOL_PICK = posx(424.62, 224.29, 88.75, P_SCAN[3], P_SCAN[4], P_SCAN[5])

        movel(P_TOOL_UP, vel=VELOCITY, acc=ACC)
        movel(P_TOOL_PICK, vel=VELOCITY, acc=ACC)

        gripper.set_target_width(DRIVER_OPEN_WIDTH)
        gripper.open_gripper()
        wait(1.0)

        movel(P_TOOL_UP, vel=VELOCITY, acc=ACC)
        self.get_logger().info("🏁 드라이버 반납 완료")

    def screw_motion_abs(self, up_pose: List[float], down_pose: List[float], is_unscrew: bool = True):
        """
        up_pose / down_pose: [x,y,z,rx,ry,rz] 절대좌표
        - up 이동 → down 이동 → 회전 → up 이동 → 역회전
        """
        from DSR_ROBOT2 import posx, wait, movel, DR_MV_MOD_REL

        if len(up_pose) != 6 or len(down_pose) != 6:
            raise ValueError("pose must be length 6")

        movel(posx(*up_pose), vel=VELOCITY, acc=ACC)
        movel(posx(*down_pose), vel=60, acc=60)

        direction = -30 if is_unscrew else 30
        P_ROT = posx(0, 0, 0, 0, 0, direction)

        self.get_logger().info(f"🔧 나사 {'풀기' if is_unscrew else '조이기'} 회전 중...")
        for _ in range(8):
            movel(P_ROT, time=0.5, mod=DR_MV_MOD_REL)
        wait(0.5)

        movel(posx(*up_pose), vel=VELOCITY, acc=ACC)
        P_ROT_BACK = posx(0, 0, 0, 0, 0, -direction)

        self.get_logger().info("🔄 케이블 꼬임 방지 역회전 중...")
        for _ in range(8):
            movel(P_ROT_BACK, time=0.5, mod=DR_MV_MOD_REL)
        wait(0.5)

    def pick_driver_from_xyzt(self, tx: float, ty: float, tz: float):
        """RealSense 기반으로 계산된 robot_x/y/z로 드라이버 픽업(간단 상향 접근)."""
        from DSR_ROBOT2 import posx, movel, wait

        # open
        gripper.set_target_width(DRIVER_OPEN_WIDTH)
        gripper.open_gripper()
        wait(0.5)

        z_approach = float(tz) + float(DRIVER_APPROACH_DZ)
        z_grasp = float(tz) + float(DRIVER_GRASP_DZ)

        P_UP = posx(float(tx), float(ty), float(z_approach), DRIVER_RX, DRIVER_RY, DRIVER_RZ)
        P_GRASP = posx(float(tx), float(ty), float(z_grasp), DRIVER_RX, DRIVER_RY, DRIVER_RZ)

        movel(P_UP, vel=VELOCITY, acc=ACC)
        movel(P_GRASP, vel=40, acc=40)

        gripper.set_target_width(DRIVER_GRIP_WIDTH)
        gripper.close_gripper()
        wait(1.0)

        movel(P_UP, vel=VELOCITY, acc=ACC)
        self.get_logger().info("🪛 감지된 드라이버 픽업 완료")

    def handover_driver_home(self):
        """P_SCAN으로 가서 드라이버를 사람에게 전달(그리퍼 오픈)."""
        from DSR_ROBOT2 import posx, movel, wait

        movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
        gripper.set_target_width(DRIVER_OPEN_WIDTH)
        gripper.open_gripper()
        wait(1.0)
        self.get_logger().info("🤝 드라이버 전달(그리퍼 오픈) 완료")

    # =========================
    # Job 실행
    # =========================
    def execute_job(self, job_id: int, tasks: List[dict]) -> bool:
        """성공 True / 실패 False."""
        from DSR_ROBOT2 import posx, movel, wait

        for task in sorted(tasks, key=lambda x: x.get("step", 999)):
            if not rclpy.ok():
                return False

            while self.paused and rclpy.ok():
                time.sleep(0.2)
            if not rclpy.ok():
                return False

            action = str(task.get("action", "")).strip()
            label = str(task.get("label", "unknown")).strip()

            # 1) 홈 이동
            if action == "prepare":
                movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)

            # 2) 홈에서 대기
            elif action == "wait":
                sec = float(task.get("wait_sec", UNKNOWN_WAIT_SEC))
                movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
                self.get_logger().info(f"⏳ P_SCAN 대기 {sec:.1f}s (label={label})")
                wait(sec)

            # 3) 드라이버 픽업/반납
            elif action == "pick_screwdriver":
                self.pickup_screwdriver()

            elif action == "return_screwdriver":
                self.return_screwdriver()

            # 4) 나사 작업(절대좌표 매핑)
            elif action in ("unscrew", "screw"):
                key = label
                if key in ABS_POSE_MAP:
                    up_pose, down_pose = ABS_POSE_MAP[key]
                    self.screw_motion_abs(up_pose, down_pose, is_unscrew=(action == "unscrew"))
                else:
                    # 요구사항: 그 외 라벨이면 P_SCAN에서 20초 대기
                    self.get_logger().warn(f"⚠️ 절대좌표 매핑 없음({label}) → P_SCAN {UNKNOWN_WAIT_SEC}s 대기")
                    movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
                    wait(float(UNKNOWN_WAIT_SEC))

            # 5) 드라이버 전달(감지 기반)
            elif action == "pick_driver":
                tx, ty, tz = task.get("robot_x"), task.get("robot_y"), task.get("robot_z")
                if not (_is_num(tx) and _is_num(ty) and _is_num(tz)):
                    self.get_logger().warn(f"⚠️ driver 좌표 없음(robot_x/y/z) → pick_driver 스킵")
                    movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
                    wait(1.0)
                else:
                    self.pick_driver_from_xyzt(float(tx), float(ty), float(tz))

            elif action == "handover_driver_home":
                self.handover_driver_home()

            else:
                self.get_logger().warn(f"⚠️ 알 수 없는 action: {action} (label={label})")

        # Job 정상 완료
        movel(posx(P_SCAN), vel=VELOCITY, acc=ACC)
        self.get_logger().info(f"✅ Job {job_id} 완료")

        try:
            self.job_done_pub.publish(String(
                data=json.dumps({"job_id": int(job_id), "state": "done"}, ensure_ascii=False)
            ))
        except Exception as e:
            self.get_logger().warn(f"job_done publish 실패: {e}")

        return True


def perform_task_loop(node: MuscleNode):
    while rclpy.ok():
        if node.paused:
            time.sleep(0.2)
            continue

        if not node._try_init_dsr():
            time.sleep(0.2)
            continue

        job_id: Optional[int] = None
        tasks: Optional[List[dict]] = None

        with node._lock:
            if node._executing_job_id is None and node._queue:
                job_id = node._queue.popleft()
                node._executing_job_id = job_id
            if job_id is not None:
                tasks = node._jobs.get(job_id)

        if job_id is None or not tasks:
            with node._lock:
                node._executing_job_id = None
            time.sleep(0.05)
            continue

        if job_id == node.last_executed_job_id:
            with node._lock:
                node._executing_job_id = None
            time.sleep(0.05)
            continue

        with node._lock:
            node._retry_count[job_id] = node._retry_count.get(job_id, 0) + 1
            attempt = node._retry_count[job_id]

        node.get_logger().info(
            f"🎯 Job 실행 시작: job_id={job_id}, steps={len(tasks)} (attempt {attempt}/{node.max_retry})"
        )

        success = False
        try:
            success = node.execute_job(job_id, tasks)
        except Exception as e:
            node.get_logger().error(f"❌ Job 실행 중 예외: job_id={job_id}, err={e}")
            success = False

        if success:
            with node._lock:
                node.last_executed_job_id = job_id
                node._jobs.pop(job_id, None)
                node._executing_job_id = None
                node._retry_count.pop(job_id, None)

            if node.reset_after_job:
                node._soft_reset("after job")

            time.sleep(0.05)
            continue

        # 실패
        with node._lock:
            node._executing_job_id = None
            attempt = node._retry_count.get(job_id, 1)

        if attempt < node.max_retry:
            node.get_logger().warn(f"🔁 Job 재시도 예정: job_id={job_id} (next attempt {attempt+1}/{node.max_retry})")
            if node.reset_on_fail:
                node._soft_reset("fail -> retry")
            with node._lock:
                if job_id not in node._queue:
                    node._queue.appendleft(job_id)
            time.sleep(max(0.05, node.retry_delay))
            continue

        node.get_logger().error(
            f"🧯 최대 재시도 초과: job_id={job_id}. 자동 재시도 중단."
        )
        if node.reset_on_fail:
            node._soft_reset("fail -> stop")

        node.paused = True
        time.sleep(0.2)


def main(args=None):
    rclpy.init(args=args)
    node = MuscleNode()

    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    # ImportError 빠르게 확인
    from DSR_ROBOT2 import get_current_posx, movel, wait, DR_MV_MOD_REL  # noqa: F401
    from DR_common2 import posx  # noqa: F401

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    th = threading.Thread(target=perform_task_loop, args=(node,), daemon=True)
    th.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
