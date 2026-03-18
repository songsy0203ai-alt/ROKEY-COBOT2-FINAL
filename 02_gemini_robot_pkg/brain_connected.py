#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brain_workflow_final.py
- 요구사항 반영(고정 시퀀스 + 사람 작업 확인 + 드라이버(OBB) 전달)

[핵심 변경]
1) 고정 시퀀스(자동 진행)
   - relay/timer(지정 라벨) : "나사 풀기(unscrew) → (사람 작업 구간) → 나사 조이기(screw)"
   - POWER/PB/L : P_SCAN 이동 후 10초 대기 (PB/L은 대기 후 작업완료 음성 확인까지)
   - 그 외 라벨(예외): P_SCAN 이동 후 20초 대기

2) PB/L 사람 작업 확인
   - L1(1,2), L2(1,2), L3(1,2), PB1(1,2), PB2(1,2), PB3(1,2)
   - P_SCAN에서 "작업 완료되면 '작업 완료'라고 말해 주세요" 안내
   - 사용자가 "응 작업했어 / 작업 완료했어 / 완료 / 끝" 등 말하면 다음 단계 자동 진행

3) "드라이버 전달" 명령
   - Eye가 퍼블리시하는 OBB 드라이버 중심좌표(RealSense + Webcam)를 받아
     RealSense 좌표(point)를 기반으로 pick → P_SCAN에서 handover
   - 두 카메라 모두 최근에 감지된 경우에만 진행(안전)
   - 필요 토픽:
       /eye/driver_rs_center   (std_msgs/String, JSON: {"found":true,"point":[ny,nx],"score":0.9,"t":unix})
       /eye/driver_web_center  (std_msgs/String, JSON: ...)

입출력
- Sub:
   /eye/terminal_centers (std_msgs/String)  # 스캔(디버그/라벨 확인용)
   /eye/driver_rs_center (std_msgs/String)
   /eye/driver_web_center(std_msgs/String)
   /ear/speech_text      (std_msgs/String)
   /muscle/job_done      (std_msgs/String)  # {"job_id":..., "state":"done"}
- Pub:
   /brain/normalized_coords (std_msgs/String)  # Nerve로 전달될 task list
   /mouth/speech_text       (std_msgs/String)  # TTS(선택)

Env
- BRAIN_ENABLE_TTS=1/0
- BRAIN_SCAN_SEC=15
- BRAIN_DRIVER_RECENT_SEC=1.5
- BRAIN_DRIVER_WAIT_SEC=6.0  # 드라이버 감지 대기 최대 시간
"""

import os
import json
import time
import threading
import re
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


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
    """
    사람 작업 완료 발화 감지.
    - 부정 표현(아직/안했/못했/아니) 우선 차단
    - 그 다음 긍정 표현(작업완료/완료/끝/다했어/했어/네/응 등) 인식
    """
    t = _normalize_text(text)

    # negative first
    if _contains_any(t, ["아직", "안했", "못했", "아니", "잠깐", "기다", "노", "no"]):
        return False

    # positive patterns
    if _contains_any(t, ["작업완료", "완료했", "완료", "끝", "끝났", "다했", "다함", "했어", "했습니다", "해써", "오케", "ok", "응", "네", "예"]):
        return True

    return False


def _parse_command(text: str) -> Optional[str]:
    """
    명령 파싱
    - START: 작업 시작
    - DRIVER: 드라이버 전달
    - PAUSE / RESUME
    - STATUS
    - RESCAN
    - FORCE_NEXT (강제 진행/스킵)
    """
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


# =========================
# Brain Node
# =========================
class BrainNode(Node):
    # --- 고정 시퀀스 정의 ---
    RELAY_TIMER_ORDER = [
        "relay 8", "relay 3", "relay 4", "relay 5", "relay 6", "relay 7",
        "timer 8", "timer 2", "timer 7",
    ]

    WAIT10_ORDER = [
        "POWER(1)", "POWER(2)",
        "PB3(1)", "PB3(2)",
        "PB1(1)", "PB1(2)",
        "PB2(1)", "PB2(2)",
        "L1(1)", "L2(1)", "L3(1)",
        "L1(2)", "L2(2)", "L3(2)",
    ]

    HUMAN_CONFIRM_LABELS = {
        "PB1(1)", "PB1(2)", "PB2(1)", "PB2(2)", "PB3(1)", "PB3(2)",
        "L1(1)", "L1(2)", "L2(1)", "L2(2)", "L3(1)", "L3(2)",
    }

    def __init__(self):
        super().__init__("brain_node")

        # QoS latched publish (optional)
        self.enable_latched_qos = os.getenv("BRAIN_LATCHED_QOS", "1").strip().lower() not in ("0", "false", "no")
        self.qos_latched: Optional[QoSProfile] = None
        if self.enable_latched_qos:
            self.qos_latched = QoSProfile(depth=1)
            self.qos_latched.history = HistoryPolicy.KEEP_LAST
            self.qos_latched.reliability = ReliabilityPolicy.RELIABLE
            self.qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # TTS
        self.enable_tts = os.getenv("BRAIN_ENABLE_TTS", "1").strip().lower() not in ("0", "false", "no")
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

        # Scan
        self.is_scanning = True
        self.scan_duration = float(os.getenv("BRAIN_SCAN_SEC", "15.0"))
        self.start_time = self.get_clock().now()
        self.object_db: Dict[str, List[int]] = {}

        # Workflow state
        self.running = False
        self.paused = False

        self.plan: List[Dict[str, Any]] = []
        self.plan_idx: int = 0

        self.waiting_human_done = False
        self.waiting_human_label: Optional[str] = None
        self._human_done_early = False
        self._last_human_prompt_time = time.time()

        # Job sync
        self.job_id = int(time.time() * 1000) % 1000000000
        self.awaiting_job_completion = False
        self.awaiting_job_id: Optional[int] = None
        self._job_sent_time = time.time()

        # Driver detection cache
        self.driver_recent_sec = float(os.getenv("BRAIN_DRIVER_RECENT_SEC", "1.5"))
        self.driver_wait_sec = float(os.getenv("BRAIN_DRIVER_WAIT_SEC", "6.0"))

        self._driver_rs: Optional[Tuple[List[int], float, float]] = None  # (point, score, t)
        self._driver_web: Optional[Tuple[List[int], float, float]] = None

        # Timer loop (scan status + periodic prompts)
        self.timer = self.create_timer(0.5, self._tick)

        self.get_logger().info("✅ Brain workflow node up.")

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
        # 스캔/디버그용 라벨 수집
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
        if not data:
            return
        if not data.get("found", False):
            return
        pt = data.get("point", None)
        if not (isinstance(pt, list) and len(pt) == 2):
            return
        score = float(data.get("score", 0.0))
        t = float(data.get("t", time.time()))
        self._driver_rs = ([int(pt[0]), int(pt[1])], score, t)

    def _on_driver_web(self, msg: String):
        data = _parse_json_safe(msg.data)
        if not data:
            return
        if not data.get("found", False):
            return
        pt = data.get("point", None)
        if not (isinstance(pt, list) and len(pt) == 2):
            return
        score = float(data.get("score", 0.0))
        t = float(data.get("t", time.time()))
        self._driver_web = ([int(pt[0]), int(pt[1])], score, t)

    # -----------------
    # Ear / Commands
    # -----------------
    def _on_ear(self, msg: String):
        raw = msg.data or ""
        cmd = _parse_command(raw)

        # 스캔 중에는 START/DRIVER는 막고, PAUSE/STATUS 정도만 허용
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
            # 재개 시, 대기 중이면 이어서 진행
            if was_paused:
                self._try_continue_after_pause()
            return

        # 사람이 해야 하는 작업(WAIT+require_confirm)에서, 로봇이 10초 대기 중일 때
        # 사용자가 먼저 '작업 완료'를 말하면 기억해두었다가 job_done 직후 자동 진행합니다.
        if self.running and self.awaiting_job_completion and (self.plan_idx < len(self.plan)):
            step = self.plan[self.plan_idx]
            if step.get('kind') == 'WAIT' and step.get('require_confirm', False):
                if _is_human_done(raw):
                    self._human_done_early = True
                    self._speak('확인했습니다. 잠시 후 다음 단계로 진행합니다.')
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
            if self.running or self.awaiting_job_completion:
                self._speak("현재 작업 중입니다. 작업이 끝난 뒤 다시 말해 주세요.")
                return
            th = threading.Thread(target=self._run_driver_delivery, daemon=True)
            th.start()
            return

        # 작업 시작
        if cmd == "START":
            if self.running:
                self._speak("이미 작업 중입니다.")
                return
            self._start_workflow()
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

        # 다음 흐름
        if not self.running:
            return

        if self.plan_idx >= len(self.plan):
            return

        step = self.plan[self.plan_idx]

        # 사람 확인이 필요한 step이면:
        # - 사용자가 대기 중(10초) 미리 '작업 완료'를 말했으면 즉시 다음 단계로
        # - 아니면 질문 상태로 전환하여 응답 대기
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
            # pause 상태면 재개까지 멈춤
            return
        self._advance_and_start_next()

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
    # Workflow control
    # -----------------
    def _start_scan(self):
        self.is_scanning = True
        self.object_db = {}
        self.start_time = self.get_clock().now()

    def _build_plan(self) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []

        # 1) relay/timer unscrew
        for lab in self.RELAY_TIMER_ORDER:
            plan.append({"kind": "UNSCREW", "label": lab})

        # 2) wait10 group (POWER/PB/L)
        for lab in self.WAIT10_ORDER:
            if lab in self.HUMAN_CONFIRM_LABELS:
                plan.append({"kind": "WAIT", "label": lab, "wait_sec": 10, "require_confirm": True})
            else:
                plan.append({"kind": "WAIT", "label": lab, "wait_sec": 10, "require_confirm": False})

        # 3) relay/timer screw
        for lab in self.RELAY_TIMER_ORDER:
            plan.append({"kind": "SCREW", "label": lab})

        return plan

    def _start_workflow(self):
        self.plan = self._build_plan()
        self.plan_idx = 0
        self.running = True
        self.paused = False
        self.waiting_human_done = False
        self.waiting_human_label = None

        self._speak("작업을 시작합니다.")
        self._start_current_step()

    def _try_continue_after_pause(self):
        # pause 해제 후 이어서
        if not self.running:
            return
        if self.awaiting_job_completion:
            return
        if self.waiting_human_done:
            # 사람 답변 기다리는 상태 그대로
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
            if step.get("require_confirm", False):
                self._human_done_early = False
            # 사람이 해야 하는 작업이면: 먼저 10초 대기(작업 시간) → job_done 후 질문
            self._speak(f"{label} 작업을 진행해 주세요. {wait_sec}초 대기합니다.")
            tasks = self._make_home_wait_job(label=label, wait_sec=wait_sec)
            self._publish_tasks(tasks)
            self._mark_job_sent(tasks)
            return

        # unknown → 20초 대기
        self._speak(f"{label} 단계는 자동 처리 대상이 아닙니다. P_SCAN에서 20초 대기합니다.")
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
    # Driver delivery
    # -----------------
    def _driver_is_recent(self, item: Optional[Tuple[List[int], float, float]]) -> bool:
        if not item:
            return False
        _pt, _score, t = item
        return (time.time() - float(t)) <= self.driver_recent_sec

    def _run_driver_delivery(self):
        # 최신 detection이 들어올 때까지 최대 driver_wait_sec 대기
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
        self._mark_job_sent(tasks)

    # -----------------
    # Task builders
    # -----------------
    def _next_job_id(self) -> int:
        self.job_id += 1
        return self.job_id

    def _make_screw_job(self, label: str, is_unscrew: bool) -> List[dict]:
        jid = self._next_job_id()
        action = "unscrew" if is_unscrew else "screw"

        # point는 없어도 됨(특정 라벨은 muscle이 절대좌표로 처리)
        pt = self.object_db.get(label)

        tasks: List[dict] = [
            {"step": 1, "action": "pick_screwdriver", "label": "screwdriver", "execute": True, "job_id": jid},
            {
                "step": 2,
                "action": action,
                "label": label,
                "terminal": label,
                "point": pt,  # 있을 때만 들어감(None 가능)
                "execute": True,
                "job_id": jid,
            },
            {"step": 3, "action": "return_screwdriver", "label": "screwdriver", "execute": True, "job_id": jid},
            {"step": 4, "action": "prepare", "label": "prepare", "execute": True, "job_id": jid},
        ]
        return tasks

    def _make_home_wait_job(self, label: str, wait_sec: int) -> List[dict]:
        jid = self._next_job_id()
        tasks: List[dict] = [
            {"step": 1, "action": "prepare", "label": "prepare", "execute": True, "job_id": jid},
            {"step": 2, "action": "wait", "label": label, "wait_sec": int(wait_sec), "execute": True, "job_id": jid},
        ]
        return tasks

    def _make_driver_delivery_job(self, rs_point: List[int]) -> List[dict]:
        jid = self._next_job_id()
        tasks: List[dict] = [
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
        return tasks

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

    def _mark_job_sent(self, tasks: List[dict]):
        if not tasks:
            return
        try:
            self.awaiting_job_completion = True
            self.awaiting_job_id = int(tasks[0].get("job_id", 0))
            self._job_sent_time = time.time()
        except Exception:
            pass

    # -----------------
    # Status
    # -----------------
    def _status_text(self) -> str:
        if self.is_scanning:
            return "상태: 스캔 중"
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

