#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_workflow_final_full.py
- brain_workflow_final.py(고정 워크플로우)를 기반으로,
  brain(4).py의 "스캔 + 회로도 기반 순서 추론"(Gemini) 기능을 통합한 최종본.

✅ 주요 기능
1) 스캔: /eye/terminal_centers(JSON list)를 일정 시간(BRAIN_SCAN_SEC) 수집해서 object_db(label->point) 구성
2) 작업 시작(음성): '작업 시작' → (선택) Gemini로 회로도 기반 라벨 순서 추론 → 그 순서대로 워크플로우 실행
3) 워크플로우 규칙(라벨별)
   - relay*/timer*: UNSCREW → (사람 작업 구간 WAIT) → SCREW
   - POWER/PB/L: P_SCAN 위치에서 WAIT(10초), PB/L은 WAIT 후 '작업 완료' 음성 확인
   - 기타: WAIT(20초)
4) 사람 작업 완료 확인: /ear/speech_text에서 "작업 완료" 유사 발화 감지
5) 드라이버 전달: "드라이버 전달/가져와" → /eye/driver_rs_center + /eye/driver_web_center 최근 감지 시 pick→handover

토픽
- Sub:
  /eye/terminal_centers (std_msgs/String)  # JSON list: [{"label":"PB1(1)","point":[ny,nx]}, ...]
  /eye/driver_rs_center (std_msgs/String) # JSON: {"found":true,"point":[ny,nx],"score":0.9,"t":unix}
  /eye/driver_web_center(std_msgs/String) # JSON: {"found":true,"point":[ny,nx],"score":0.9,"t":unix}
  /ear/speech_text      (std_msgs/String)
  /muscle/job_done      (std_msgs/String) # JSON: {"job_id":123,"state":"done"}
- Pub:
  /brain/normalized_coords (std_msgs/String)  # JSON list tasks -> Nerve/Muscle
  /mouth/speech_text       (std_msgs/String)  # TTS(선택)

환경변수
- BRAIN_ENABLE_TTS=1/0
- BRAIN_SCAN_SEC=15

- BRAIN_USE_GEMINI_ORDER=1/0
- GEMINI_API_KEY_BRAIN=...
- BRAIN_MODEL_ID=gemini-robotics-er-1.5-preview
- BRAIN_RESOURCE_DIR=~/cobot_ws/src/cobot2_ws/gemini_robot_pkg/resource

- BRAIN_RELAY_WORK_WAIT_SEC=15
- BRAIN_RELAY_WORK_CONFIRM=1/0

- BRAIN_DRIVER_RECENT_SEC=1.5
- BRAIN_DRIVER_WAIT_SEC=6.0

비고
- Gemini 사용이 꺼져있거나(API 키/라이브러리 없음 포함) 추론 실패 시, 고정 순서(RELAY_TIMER_ORDER + WAIT10_ORDER)로 실행
"""

import os
import re
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Gemini (optional)
try:
    import PIL.Image
    from google import genai
    from google.genai import types
except Exception:
    PIL = None  # type: ignore
    genai = None  # type: ignore
    types = None  # type: ignore


# =========================
# Utils
# =========================

def _normalize_text(s: str) -> str:
    return (s or "").replace(" ", "").replace(".", "").replace("!", "").replace("?", "").strip().lower()


def _contains_any(s: str, keywords: List[str]) -> bool:
    return any(k in s for k in keywords)


def _parse_json_safe(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _is_human_done(text: str) -> bool:
    """사람 작업 완료 발화 감지."""
    t = _normalize_text(text)

    # negative first
    if _contains_any(t, ["아직", "안했", "못했", "아니", "잠깐", "기다", "노", "no"]):
        return False

    # positive patterns
    if _contains_any(
        t,
        [
            "작업완료",
            "완료했",
            "완료",
            "끝",
            "끝났",
            "다했",
            "다함",
            "했어",
            "했습니다",
            "해써",
            "오케",
            "ok",
            "응",
            "네",
            "예",
        ],
    ):
        return True

    return False


def _parse_command(text: str) -> Optional[str]:
    """음성 명령 파싱."""
    raw = text or ""
    t = _normalize_text(raw)

    # start
    if ("작업" in t and "시작" in t) or t == "시작":
        return "START"

    # driver
    if ("드라이버" in t) and _contains_any(t, ["전달", "갖다", "가져", "줘", "주라", "주세요"]):
        return "DRIVER"

    # pause/resume
    if _contains_any(t, ["멈춰", "정지", "스톱", "stop"]):
        return "PAUSE"
    if _contains_any(t, ["재개", "계속", "resume"]):
        return "RESUME"

    # status
    if _contains_any(t, ["상태", "status"]):
        return "STATUS"

    # rescan
    if ("스캔" in t) and _contains_any(t, ["다시", "재"]):
        return "RESCAN"

    # force next
    if _contains_any(t, ["강제", "스킵", "skip", "넘어", "패스", "pass"]) and _contains_any(t, ["진행", "다음", "넘어", "패스", "pass"]):
        return "FORCE_NEXT"

    return None


def _get_pkg_resource_path(filename: str) -> str:
    """gemini_robot_pkg/resource 에서 회로도 이미지를 찾기."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("gemini_robot_pkg")
        p = os.path.join(share, "resource", filename)
        if os.path.exists(p):
            return p
    except Exception:
        pass

    fallback = os.path.expanduser(
        os.getenv("BRAIN_RESOURCE_DIR", "~/cobot_ws/src/cobot2_ws/gemini_robot_pkg/resource")
    )
    return os.path.join(fallback, filename)


def _extract_json_array(text: str):
    """Gemini 응답에서 JSON 배열을 최대한 안전하게 추출."""
    text = (text or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# =========================
# Brain Node
# =========================


class BrainNode(Node):
    # --- 고정 순서 (fallback) ---
    RELAY_TIMER_ORDER = [
        "relay 8",
        "relay 3",
        "relay 4",
        "relay 5",
        "relay 6",
        "relay 7",
        "timer 8",
        "timer 2",
        "timer 7",
    ]

    WAIT10_ORDER = [
        "POWER(1)",
        "POWER(2)",
        "PB3(1)",
        "PB3(2)",
        "PB1(1)",
        "PB1(2)",
        "PB2(1)",
        "PB2(2)",
        "L1(1)",
        "L2(1)",
        "L3(1)",
        "L1(2)",
        "L2(2)",
        "L3(2)",
    ]

    HUMAN_CONFIRM_LABELS = {
        "PB1(1)",
        "PB1(2)",
        "PB2(1)",
        "PB2(2)",
        "PB3(1)",
        "PB3(2)",
        "L1(1)",
        "L1(2)",
        "L2(1)",
        "L2(2)",
        "L3(1)",
        "L3(2)",
    }

    def __init__(self):
        super().__init__("brain_node")

        # -------- Gemini planning options --------
        self.use_gemini_order = os.getenv("BRAIN_USE_GEMINI_ORDER", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        self.api_key = os.getenv("GEMINI_API_KEY_BRAIN", "").strip()
        self.model_name = os.getenv("BRAIN_MODEL_ID", "gemini-robotics-er-1.5-preview")
        self.client = None

        if self.use_gemini_order:
            if (not self.api_key) or (genai is None):
                self.get_logger().warn(
                    "⚠️ BRAIN_USE_GEMINI_ORDER=1 이지만 GEMINI_API_KEY_BRAIN 또는 google-genai 라이브러리가 없습니다. 고정 순서로 진행합니다."
                )
                self.use_gemini_order = False
            else:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except Exception as e:
                    self.get_logger().warn(f"⚠️ Gemini client 생성 실패: {e} (고정 순서로 진행)")
                    self.use_gemini_order = False
                    self.client = None

        # 회로도 리소스 (Gemini가 켜져있을 때만 사용)
        self.circuit_diagram_path = _get_pkg_resource_path("plc_circuit.png")
        self.relay_diagram_path = _get_pkg_resource_path("relay_circuit.jpg")
        self.timer_diagram_path = _get_pkg_resource_path("timer_circuit.jpg")

        self.is_planning = False
        self._workflow_lock = threading.Lock()

        # relay/timer 사람 작업 구간(대기+확인)
        self.relay_work_wait_sec = int(float(os.getenv("BRAIN_RELAY_WORK_WAIT_SEC", "15")))
        self.relay_work_require_confirm = os.getenv("BRAIN_RELAY_WORK_CONFIRM", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

        # -------- QoS / TTS --------
        self.enable_latched_qos = os.getenv("BRAIN_LATCHED_QOS", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        self.qos_latched: Optional[QoSProfile] = None
        if self.enable_latched_qos:
            self.qos_latched = QoSProfile(depth=1)
            self.qos_latched.history = HistoryPolicy.KEEP_LAST
            self.qos_latched.reliability = ReliabilityPolicy.RELIABLE
            self.qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.enable_tts = os.getenv("BRAIN_ENABLE_TTS", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        self.mouth_pub = self.create_publisher(String, "/mouth/speech_text", 10) if self.enable_tts else None

        # Publishers
        if self.enable_latched_qos and self.qos_latched is not None:
            self.pub = self.create_publisher(String, "/brain/normalized_coords", self.qos_latched)
        else:
            self.pub = self.create_publisher(String, "/brain/normalized_coords", 10)

        # Subscribers
        self.sub_eye = self.create_subscription(String, "/eye/terminal_centers", self._on_eye, 10)
        self.sub_driver_rs = self.create_subscription(String, "/eye/driver_rs_center", self._on_driver_rs, 10)
        self.sub_driver_web = self.create_subscription(String, "/eye/driver_web_center", self._on_driver_web, 10)
        self.sub_ear = self.create_subscription(String, "/ear/speech_text", self._on_ear, 10)
        self.sub_job_done = self.create_subscription(String, "/muscle/job_done", self._on_job_done, 10)

        # -------- Scan --------
        self.is_scanning = True
        self.scan_duration = float(os.getenv("BRAIN_SCAN_SEC", "15.0"))
        self.start_time = self.get_clock().now()
        self.object_db: Dict[str, List[int]] = {}

        # -------- Workflow state --------
        self.running = False
        self.paused = False

        self.plan: List[Dict[str, Any]] = []
        self.plan_idx: int = 0
        self.last_plan_labels: List[str] = []

        self.waiting_human_done = False
        self.waiting_human_label: Optional[str] = None
        self._human_done_early = False
        self._last_human_prompt_time = time.time()

        # Job sync
        self.job_id = int(time.time() * 1000) % 1000000000
        self.awaiting_job_completion = False
        self.awaiting_job_id: Optional[int] = None
        self._job_sent_time = time.time()
        self._is_driver_job = False

        # Driver detection cache
        self.driver_recent_sec = float(os.getenv("BRAIN_DRIVER_RECENT_SEC", "1.5"))
        self.driver_wait_sec = float(os.getenv("BRAIN_DRIVER_WAIT_SEC", "6.0"))
        self._driver_rs: Optional[Tuple[List[int], float, float]] = None  # (point, score, t)
        self._driver_web: Optional[Tuple[List[int], float, float]] = None

        # Timer loop
        self.timer = self.create_timer(0.5, self._tick)

        self.get_logger().info("✅ Brain workflow(full) node up.")

    # -----------------
    # Speech helper
    # -----------------
    def _speak(self, text: str):
        t = (text or "").strip()
        if not t:
            return
        self.get_logger().info(f"🗣️ {t}")
        if self.enable_tts and self.mouth_pub is not None:
            try:
                self.mouth_pub.publish(String(data=t))
            except Exception as e:
                self.get_logger().warn(f"mouth publish 실패: {e}")

    # -----------------
    # Scan / Eye
    # -----------------
    def _on_eye(self, msg: String):
        try:
            dets = json.loads(msg.data)
            if not isinstance(dets, list):
                return
            if self.is_scanning:
                for item in dets:
                    label = str(item.get("label", "")).strip()
                    pt = item.get("point", None)
                    if label and isinstance(pt, list) and len(pt) == 2:
                        self.object_db[label] = [int(pt[0]), int(pt[1])]
        except Exception:
            return

    def _on_driver_rs(self, msg: String):
        data = _parse_json_safe(msg.data)
        if not data or (not data.get("found", False)):
            return
        pt = data.get("point", None)
        if not (isinstance(pt, list) and len(pt) == 2):
            return
        score = float(data.get("score", 0.0))
        t = float(data.get("t", time.time()))
        self._driver_rs = ([int(pt[0]), int(pt[1])], score, t)

    def _on_driver_web(self, msg: String):
        data = _parse_json_safe(msg.data)
        if not data or (not data.get("found", False)):
            return
        pt = data.get("point", None)
        if not (isinstance(pt, list) and len(pt) == 2):
            return
        score = float(data.get("score", 0.0))
        t = float(data.get("t", time.time()))
        self._driver_web = ([int(pt[0]), int(pt[1])], score, t)

    # -----------------
    # Tick
    # -----------------
    def _tick(self):
        # scan 진행
        if self.is_scanning:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed >= self.scan_duration:
                self.is_scanning = False
                self._speak("스캔 완료. 작업 시작이라고 말하면 시작합니다.")
                self.get_logger().info(f"스캔 라벨 수: {len(self.object_db)}")
            return

        # 사람 작업 완료 대기 중이면, 주기적으로 리마인드
        if self.waiting_human_done and (not self.paused):
            if time.time() - self._last_human_prompt_time > 20.0:
                lab = self.waiting_human_label or ""
                self._speak(f"{lab} 작업 완료되었나요? 완료했으면 '작업 완료'라고 말해 주세요.")
                self._last_human_prompt_time = time.time()

    # -----------------
    # Ear / Commands
    # -----------------
    def _on_ear(self, msg: String):
        raw = msg.data or ""
        cmd = _parse_command(raw)
        if not cmd:
            return

        # 스캔 중에는 START/DRIVER는 막고, PAUSE/STATUS/RESCAN만 허용
        if self.is_scanning and cmd not in ("PAUSE", "STATUS", "RESCAN"):
            self._speak("스캔 중입니다. 잠시만 기다려 주세요.")
            return

        # 상태
        if cmd == "STATUS":
            self._speak(self._status_text())
            return

        # 재스캔
        if cmd == "RESCAN":
            self._start_scan()
            self._speak("스캔을 다시 시작합니다.")
            return

        # 정지/재개
        if cmd == "PAUSE":
            self.paused = True
            self._speak("일시 정지")
            return

        if cmd == "RESUME":
            was_paused = self.paused
            self.paused = False
            self._speak("재개합니다.")
            if was_paused:
                self._try_continue_after_pause()
            return

        # 사람이 해야 하는 WAIT(require_confirm)에서 로봇이 대기하는 동안,
        # 사용자가 먼저 '작업 완료'를 말하면 job_done 직후 자동 진행하도록 표시
        if self.running and self.awaiting_job_completion and (self.plan_idx < len(self.plan)):
            step = self.plan[self.plan_idx]
            if step.get("kind") == "WAIT" and step.get("require_confirm", False):
                if _is_human_done(raw):
                    self._human_done_early = True
                    self._speak("확인했습니다. 잠시 후 다음 단계로 진행합니다.")
                    return

        # 사람 작업 완료 대기 중이면 → 완료 발화 처리
        if self.waiting_human_done:
            if _is_human_done(raw):
                self._handle_human_done()
            elif cmd == "FORCE_NEXT":
                self._speak("강제 진행합니다.")
                self._handle_human_done(force=True)
            return

        # 강제 다음(사람 대기가 아닐 때)
        if cmd == "FORCE_NEXT":
            if self.running and (not self.awaiting_job_completion) and (not self.paused):
                self._speak("강제 다음 단계로 넘어갑니다.")
                self._advance_and_start_next()
            else:
                self._speak("지금은 강제 진행할 수 없습니다.")
            return

        # 드라이버 전달
        if cmd == "DRIVER":
            if self.awaiting_job_completion or self.is_planning:
                self._speak("로봇이 동작 중입니다. 잠시 후 다시 말해 주세요.")
                return
            th = threading.Thread(target=self._run_driver_delivery, daemon=True)
            th.start()
            return

        # 작업 시작
        if cmd == "START":
            if self.running:
                self._speak("이미 작업 중입니다.")
                return
            if self.is_planning:
                self._speak("이미 순서를 계산 중입니다. 잠시만 기다려 주세요.")
                return
            th = threading.Thread(target=self._start_workflow_thread, daemon=True)
            th.start()
            return

    # -----------------
    # Job done
    # -----------------
    def _on_job_done(self, msg: String):
        jid = None
        state = ""
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                jid = int(data.get("job_id", -1))
                state = str(data.get("state", "")).strip().lower()
        except Exception:
            return

        if state != "done":
            return

        if not self.awaiting_job_completion:
            return

        if self.awaiting_job_id is not None and jid != self.awaiting_job_id:
            return

        # job done 인정
        self.awaiting_job_completion = False
        self.awaiting_job_id = None

        if getattr(self, "_is_driver_job", False):
            self._is_driver_job = False
            self._speak("드라이버 전달을 완료했습니다. 하던 작업을 계속해 주세요.")
            return

        if not self.running:
            return

        if self.plan_idx >= len(self.plan):
            return

        step = self.plan[self.plan_idx]

        # WAIT + require_confirm: job_done 이후 질문/확인
        if step.get("kind") == "WAIT" and step.get("require_confirm", False):
            if getattr(self, "_human_done_early", False):
                self._human_done_early = False
                if self.paused:
                    return
                self._advance_and_start_next()
                return

            self.waiting_human_done = True
            self.waiting_human_label = str(step.get("label", ""))
            self._last_human_prompt_time = time.time()
            self._speak(f"{self.waiting_human_label} 작업이 완료되었나요? 완료했으면 '작업 완료'라고 말해 주세요.")
            return

        # 그 외는 자동 다음
        if self.paused:
            return
        self._advance_and_start_next()

    # -----------------
    # Workflow control
    # -----------------
    def _start_scan(self):
        self.is_scanning = True
        self.object_db = {}
        self.start_time = self.get_clock().now()

    def _start_workflow_thread(self):
        with self._workflow_lock:
            if self.running or self.is_planning:
                return
            self.is_planning = True

        try:
            # 계획 만들기
            labels: List[str] = []
            if self.use_gemini_order:
                self._speak("회로도 기반 작업 순서를 계산 중입니다.")
                labels = self._ask_gemini_for_terminal_order()

            if labels:
                self.last_plan_labels = labels
                self.plan = self._build_plan_from_labels(labels)
                self.plan_idx = 0
            else:
                self.last_plan_labels = []
                self.plan = self._build_plan_fallback_fixed()
                self.plan_idx = 0
                self._speak("고정 순서로 작업을 진행합니다.")

            # 실행
            self.running = True
            self.paused = False
            self.waiting_human_done = False
            self.waiting_human_label = None

            self._speak("작업을 시작합니다.")
            self._start_current_step()

        finally:
            with self._workflow_lock:
                self.is_planning = False

    def _announce_order(self, labels: List[str]):
        if not labels:
            return
        # TTS가 너무 길어지지 않게 10개씩 끊어서 말하기
        chunk = 10
        total = len(labels)
        self._speak(f"총 {total}개 라벨을 회로 순서로 정렬했습니다.")
        for i in range(0, min(total, 30), chunk):
            part = labels[i : i + chunk]
            self._speak("작업 순서: " + " -> ".join(part))
        if total > 30:
            self._speak("이후 순서는 로그에서 확인할 수 있습니다.")

    def _is_relay_timer_label(self, label: str) -> bool:
        t = (label or "").strip().lower()
        return t.startswith("relay") or t.startswith("timer")

    def _is_power_pb_l_label(self, label: str) -> bool:
        t = (label or "").strip().upper()
        return t.startswith("POWER") or t.startswith("PB") or t.startswith("L")

    def _build_plan_from_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """Gemini가 준 라벨 순서를 워크플로우(step plan)로 변환."""
        plan: List[Dict[str, Any]] = []

        # de-dup (keep order)
        seen = set()
        uniq: List[str] = []
        for lab in labels:
            if lab and lab not in seen:
                seen.add(lab)
                uniq.append(lab)

        for lab in uniq:
            if self._is_relay_timer_label(lab):
                plan.append({"kind": "UNSCREW", "label": lab})
                plan.append(
                    {
                        "kind": "WAIT",
                        "label": lab,
                        "wait_sec": int(self.relay_work_wait_sec),
                        "require_confirm": bool(self.relay_work_require_confirm),
                    }
                )
                plan.append({"kind": "SCREW", "label": lab})
                continue

            if self._is_power_pb_l_label(lab):
                req = lab in self.HUMAN_CONFIRM_LABELS
                plan.append({"kind": "WAIT", "label": lab, "wait_sec": 10, "require_confirm": req})
                continue

            # 기타
            plan.append({"kind": "WAIT", "label": lab, "wait_sec": 20, "require_confirm": False})

        return plan

    def _build_plan_fallback_fixed(self) -> List[Dict[str, Any]]:
        """Gemini 실패/비활성 시 고정 순서 플랜."""
        plan: List[Dict[str, Any]] = []

        # relay/timer: 라벨별 unscrew -> wait -> screw
        for lab in self.RELAY_TIMER_ORDER:
            plan.append({"kind": "UNSCREW", "label": lab})
            plan.append(
                {
                    "kind": "WAIT",
                    "label": lab,
                    "wait_sec": int(self.relay_work_wait_sec),
                    "require_confirm": bool(self.relay_work_require_confirm),
                }
            )
            plan.append({"kind": "SCREW", "label": lab})

        # POWER/PB/L: wait
        for lab in self.WAIT10_ORDER:
            plan.append(
                {
                    "kind": "WAIT",
                    "label": lab,
                    "wait_sec": 10,
                    "require_confirm": bool(lab in self.HUMAN_CONFIRM_LABELS),
                }
            )

        return plan

    def _try_continue_after_pause(self):
        if not self.running:
            return
        if self.awaiting_job_completion:
            return
        if self.waiting_human_done:
            return
        self._start_current_step()

    def _advance_and_start_next(self):
        self.plan_idx += 1
        if self.plan_idx >= len(self.plan):
            self.running = False
            self._speak("전체 작업이 완료되었습니다.")
            return
        self._start_current_step()

    def _start_current_step(self):
        if not self.running:
            return
        if self.paused:
            return
        if self.awaiting_job_completion:
            return
        if self.waiting_human_done:
            return

        if self.plan_idx >= len(self.plan):
            self.running = False
            self._speak("전체 작업이 완료되었습니다.")
            return

        step = self.plan[self.plan_idx]
        kind = step.get("kind")
        label = str(step.get("label", ""))

        if kind == "UNSCREW":
            self._speak(f"{label} 나사 풀기 시작")
            tasks = self._make_screw_job(label=label, is_unscrew=True)
            self._publish_tasks(tasks)
            self._mark_job_sent(tasks)
            return

        if kind == "SCREW":
            self._speak(f"{label} 나사 조이기 시작")
            tasks = self._make_screw_job(label=label, is_unscrew=False)
            self._publish_tasks(tasks)
            self._mark_job_sent(tasks)
            return

        if kind == "WAIT":
            wait_sec = int(step.get("wait_sec", 10))
            require_confirm = bool(step.get("require_confirm", False))
            if require_confirm:
                self._human_done_early = False
            self._speak(f"{label} 작업을 진행해 주세요. {wait_sec}초 대기합니다.")
            tasks = self._make_home_wait_job(label=label, wait_sec=wait_sec)
            self._publish_tasks(tasks)
            self._mark_job_sent(tasks)
            return

        # unknown
        self._speak(f"{label} 단계는 정의되지 않았습니다. P_SCAN에서 20초 대기합니다.")
        tasks = self._make_home_wait_job(label=label, wait_sec=20)
        self._publish_tasks(tasks)
        self._mark_job_sent(tasks)

    def _handle_human_done(self, force: bool = False):
        if not self.waiting_human_done:
            return

        lab = self.waiting_human_label or ""
        self.waiting_human_done = False
        self.waiting_human_label = None

        if force:
            self._speak(f"{lab} 작업을 강제로 완료 처리합니다. 다음 단계로 진행합니다.")
        else:
            self._speak(f"{lab} 작업 완료 확인. 다음 단계로 진행합니다.")

        if self.paused:
            return
        self._advance_and_start_next()

    # -----------------
    # Gemini planning
    # -----------------
    def _ask_gemini_for_terminal_order(self) -> List[str]:
        if not self.client:
            self.get_logger().warn("Gemini client 없음 (API KEY/라이브러리 확인).")
            return []
        if not self.object_db:
            self.get_logger().warn("object_db가 비었습니다. 스캔 실패 가능.")
            return []

        # 이미지 로드
        try:
            circuit_img = PIL.Image.open(self.circuit_diagram_path)  # type: ignore
            relay_img = PIL.Image.open(self.relay_diagram_path)  # type: ignore
            timer_img = PIL.Image.open(self.timer_diagram_path)  # type: ignore
        except Exception as e:
            self.get_logger().warn(f"회로도 이미지 로드 실패: {e}")
            return []

        prompt = f"""
너는 Doosan M0609 로봇의 협업 지능이야.
제공된 PLC 회로도 + 릴레이 내부 회로 + 타이머 내부 회로를 분석해서,
[환경 데이터 라벨 목록] 중에서 '작업해야 할 단자 라벨'의 순서를 결정해.

[환경 데이터 라벨 목록]:
{json.dumps(sorted(list(self.object_db.keys())), ensure_ascii=False, indent=2)}

[출력 규칙]
- 반드시 JSON 배열만 출력
- 각 원소는 아래 형식:
  {{"step": 1, "terminal": "환경 데이터에 존재하는 라벨"}}
- terminal 값은 위 라벨 목록과 **완전히 동일한 문자열**만 사용
- 회로 흐름에 따라 작업자가 따라하기 쉬운 순서로 정렬
"""

        self.get_logger().info("🧠 Gemini에게 작업 순서를 요청 중...")

        try:
            # thinking_config가 가능한 버전이면 사용
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[circuit_img, prompt, relay_img, timer_img],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=200),
                )
            )
        except Exception:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[circuit_img, prompt, relay_img, timer_img],
                config=types.GenerateContentConfig(temperature=0.0) if types else None,
            )

        txt = getattr(response, "text", "") or ""
        items = _extract_json_array(txt)
        if not items or not isinstance(items, list):
            self.get_logger().warn("Gemini 응답에서 JSON 배열을 찾지 못했습니다.")
            return []

        out: List[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            term = str(it.get("terminal", "")).strip()
            if term and term in self.object_db:
                out.append(term)

        # de-dup
        seen = set()
        uniq: List[str] = []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        self.get_logger().info(f"✅ Gemini 순서 확정: {len(uniq)}개")
        if uniq:
            self.get_logger().info("★★★ 작업 시퀀스(labels) ★★★: " + " -> ".join(uniq))
        return uniq

    # -----------------
    # Driver delivery
    # -----------------
    def _driver_is_recent(self, item: Optional[Tuple[List[int], float, float]]) -> bool:
        if not item:
            return False
        _pt, _score, t = item
        return (time.time() - float(t)) <= self.driver_recent_sec

    def _run_driver_delivery(self):
        self._speak("드라이버를 찾는 중입니다.")

        t_end = time.time() + max(1.0, self.driver_wait_sec)
        while time.time() < t_end and rclpy.ok():
            if self._driver_is_recent(self._driver_rs) and self._driver_is_recent(self._driver_web):
                break
            time.sleep(0.1)

        if not (self._driver_is_recent(self._driver_rs) and self._driver_is_recent(self._driver_web)):
            self._speak("드라이버를 찾지 못했습니다. 카메라 화면에서 드라이버가 잘 보이게 해 주세요.")
            return

        rs_pt, rs_score, _ = self._driver_rs  # type: ignore
        web_pt, web_score, _ = self._driver_web  # type: ignore
        self.get_logger().info(f"🪛 driver rs={rs_pt}({rs_score:.2f}) / web={web_pt}({web_score:.2f})")

        self._speak("드라이버를 감지했습니다. 전달하겠습니다.")
        tasks = self._make_driver_delivery_job(rs_point=rs_pt)
        self._publish_tasks(tasks)
        self._mark_job_sent(tasks, is_driver=True)

    # -----------------
    # Task builders
    # -----------------
    def _next_job_id(self) -> int:
        self.job_id += 1
        return self.job_id

    def _make_screw_job(self, label: str, is_unscrew: bool) -> List[dict]:
        jid = self._next_job_id()
        action = "unscrew" if is_unscrew else "screw"

        pt = self.object_db.get(label)  # None 가능

        return [
            {"step": 1, "action": "pick_screwdriver", "label": "screwdriver", "execute": True, "job_id": jid},
            {
                "step": 2,
                "action": action,
                "label": label,
                "terminal": label,
                "point": pt,
                "execute": True,
                "job_id": jid,
            },
            {"step": 3, "action": "return_screwdriver", "label": "screwdriver", "execute": True, "job_id": jid},
            {"step": 4, "action": "prepare", "label": "prepare", "execute": True, "job_id": jid},
        ]

    def _make_home_wait_job(self, label: str, wait_sec: int) -> List[dict]:
        jid = self._next_job_id()
        return [
            {"step": 1, "action": "prepare", "label": "prepare", "execute": True, "job_id": jid},
            {"step": 2, "action": "wait", "label": label, "wait_sec": int(wait_sec), "execute": True, "job_id": jid},
        ]

    def _make_driver_delivery_job(self, rs_point: List[int]) -> List[dict]:
        jid = self._next_job_id()
        return [
            {
                "step": 1,
                "action": "pick_driver",
                "label": "driver",
                "terminal": "driver",
                "point": rs_point,
                "execute": True,
                "job_id": jid,
            },
            {
                "step": 2,
                "action": "handover_driver_home",
                "label": "driver",
                "execute": True,
                "job_id": jid,
            },
            {"step": 3, "action": "prepare", "label": "prepare", "execute": True, "job_id": jid},
        ]

    # -----------------
    # Publish + job sync
    # -----------------
    def _publish_tasks(self, tasks: List[dict]):
        if not tasks:
            return
        try:
            self.pub.publish(String(data=json.dumps(tasks, ensure_ascii=False)))
            jid = int(tasks[0].get("job_id", 0))
            self.get_logger().info(f"📤 publish /brain/normalized_coords (job_id={jid}, steps={len(tasks)})")
        except Exception as e:
            self.get_logger().error(f"publish 실패: {e}")

    def _mark_job_sent(self, tasks: List[dict], is_driver: bool = False):
        if not tasks:
            return
        try:
            self.awaiting_job_completion = True
            self.awaiting_job_id = int(tasks[0].get("job_id", 0))
            self._job_sent_time = time.time()
            self._is_driver_job = is_driver
        except Exception:
            pass

    # -----------------
    # Status
    # -----------------
    def _status_text(self) -> str:
        if self.is_scanning:
            return "상태: 스캔 중"
        if self.is_planning:
            return "상태: 회로 순서 계산 중"
        if not self.running:
            return "상태: 대기 중"
        if self.paused:
            return "상태: 일시 정지"
        if self.awaiting_job_completion:
            return f"상태: 로봇 작업 중 (step {self.plan_idx+1}/{len(self.plan)})"
        if self.waiting_human_done:
            return f"상태: 사람 작업 완료 대기 ({self.waiting_human_label})"
        return f"상태: 진행 준비 (step {self.plan_idx+1}/{len(self.plan)})"


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()