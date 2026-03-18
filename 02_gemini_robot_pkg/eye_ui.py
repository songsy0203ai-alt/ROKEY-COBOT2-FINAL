#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""gemini_hmi_ui.py

Gemini Robotics ER - Integrated HMI UI

요구사항 반영
- ROS 2 Humble + rclpy (별도 스레드에서 spin)  ✅
- PyQt6 기반 Industrial HMI 스타일(QSS)        ✅
- 토픽 연동(필수 5개)
  1) /eye/image_raw (sensor_msgs/Image)
  2) /muscle/job_done (std_msgs/String, JSON)
  3) /nerve/robot_coords (std_msgs/String, JSON list)
  4) /ear/speech_text, /mouth/speech_text (std_msgs/String)
  5) /safety/recovery (std_msgs/Bool) - Safety Recovery 버튼

실행
- (현장) ROS 환경:
    python3 gemini_hmi_ui.py

- (ROS 없는 PC 데모):
    python3 gemini_hmi_ui.py --demo

- (이 파일만 검증/테스트: PyQt6, ROS 없이도 가능)
    python3 gemini_hmi_ui.py --selftest
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# =========================
# Optional imports (ROS / Qt)
# =========================

ROS_AVAILABLE = False
CV_BRIDGE_AVAILABLE = False

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import Bool as RosBool
    from std_msgs.msg import String as RosString

    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except Exception:
    CV_BRIDGE_AVAILABLE = False


QT_AVAILABLE = False
try:
    from PyQt6.QtCore import (
        QObject,
        QThread,
        QTimer,
        Qt,
        pyqtSignal,
        pyqtSlot,
    )
    from PyQt6.QtGui import QFont, QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QProgressBar,
        QSizePolicy,
        QSpacerItem,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


# =========================
# Utilities
# =========================

def safe_json_loads(s: str) -> Optional[Any]:
    """JSON String 토픽을 안전하게 파싱.

    - 유효하면 파싱 결과 반환
    - 실패하면 None
    """
    if s is None:
        return None
    txt = (s or "").strip()
    if not txt:
        return None

    try:
        return json.loads(txt)
    except Exception:
        return None


def now_ts_str() -> str:
    # HH:MM:SS.mmm
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%H:%M:%S", lt) + f".{ms:03d}"


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class JobDoneEvent:
    job_id: int
    state: str
    raw: Any


def parse_job_done(msg_text: str) -> Optional[JobDoneEvent]:
    data = safe_json_loads(msg_text)
    job_id = -1
    state = ""

    if isinstance(data, dict):
        try:
            job_id = int(data.get("job_id", -1))
        except Exception:
            job_id = -1
        state = str(data.get("state", "")).strip()
        return JobDoneEvent(job_id=job_id, state=state, raw=data)

    # 폴백: JSON이 아니라면 raw 문자열 그대로
    t = (msg_text or "").strip()
    if not t:
        return None
    return JobDoneEvent(job_id=-1, state=t, raw=t)


# =========================
# Qt Widgets (only if QT_AVAILABLE)
# =========================

if QT_AVAILABLE:

    class ScaledImageLabel(QLabel):
        """QLabel에 QPixmap을 넣으면 resize 시 비율 유지로 스케일."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pixmap_src: Optional[QPixmap] = None
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText("NO SIGNAL")
            self.setProperty("role", "video")

        def set_image(self, qimg: QImage):
            self._pixmap_src = QPixmap.fromImage(qimg)
            self._apply_scaled()

        def clear_image(self, text: str = "NO SIGNAL"):
            self._pixmap_src = None
            self.clear()
            self.setText(text)

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._apply_scaled()

        def _apply_scaled(self):
            if self._pixmap_src is None:
                return
            if self.width() <= 5 or self.height() <= 5:
                return
            scaled = self._pixmap_src.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


    class StatusLed(QWidget):
        """간단한 LED + 텍스트 표시(연결/상태용)."""

        def __init__(self, title: str, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._dot = QLabel()
            self._dot.setFixedSize(12, 12)
            self._dot.setProperty("role", "led")

            self._label = QLabel(title)
            self._label.setProperty("role", "led_text")

            lay = QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            lay.addWidget(self._dot)
            lay.addWidget(self._label)
            lay.addStretch(1)

            self.set_state(False)

        def set_state(self, ok: bool, color_hex: Optional[str] = None, tooltip: str = ""):
            if color_hex is None:
                color_hex = "#4CAF50" if ok else "#666666"

            self._dot.setStyleSheet(
                "QLabel {"
                f" background-color: {color_hex};"
                " border: 1px solid #111111;"
                " border-radius: 6px;"
                "}"
            )
            if tooltip:
                self.setToolTip(tooltip)


# =========================
# ROS Worker (Qt thread)
# =========================

if QT_AVAILABLE:

    class HmiRosNode(Node):
        """ROS2 Node: 토픽 수신/발행."""

        def __init__(
            self,
            *,
            image_topic: str,
            job_done_topic: str,
            coords_topic: str,
            ear_topic: str,
            mouth_topic: str,
            safety_topic: str,
            on_image,
            on_job_done,
            on_coords,
            on_ear,
            on_mouth,
            log_info,
        ):
            super().__init__("gemini_hmi_ui")

            self._log_info = log_info
            self._bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None

            self._sub_img = self.create_subscription(RosImage, image_topic, on_image, 10)
            self._sub_job = self.create_subscription(RosString, job_done_topic, on_job_done, 10)
            self._sub_coords = self.create_subscription(RosString, coords_topic, on_coords, 10)
            self._sub_ear = self.create_subscription(RosString, ear_topic, on_ear, 10)
            self._sub_mouth = self.create_subscription(RosString, mouth_topic, on_mouth, 10)

            self._pub_safety = self.create_publisher(RosBool, safety_topic, 10)

            self._log_info(
                f"ROS subscriptions ready: {image_topic}, {job_done_topic}, {coords_topic}, {ear_topic}, {mouth_topic}"
            )
            self._log_info(f"ROS publisher ready: {safety_topic}")

        @property
        def bridge(self) -> Optional[CvBridge]:
            return self._bridge

        def publish_safety_recovery(self, value: bool = True):
            msg = RosBool()
            msg.data = bool(value)
            self._pub_safety.publish(msg)


    class RosWorker(QObject):
        """Qt Worker living in a QThread.
        - QTimer로 rclpy executor.spin_once()를 주기 호출
        - UI 스레드로는 signal로만 데이터 전달
        """

        sig_ros_status = pyqtSignal(bool, str)  # ok, message
        sig_image = pyqtSignal(QImage)
        sig_job_done = pyqtSignal(int, str, object)  # job_id, state, raw
        sig_coords = pyqtSignal(object)  # parsed json or raw
        sig_log = pyqtSignal(str, str)  # source, text

        def __init__(
            self,
            *,
            image_topic: str = "/eye/image_raw",
            job_done_topic: str = "/muscle/job_done",
            coords_topic: str = "/nerve/robot_coords",
            ear_topic: str = "/ear/speech_text",
            mouth_topic: str = "/mouth/speech_text",
            safety_topic: str = "/safety/recovery",
            spin_hz: int = 100,
            parent: Optional[QObject] = None,
        ):
            super().__init__(parent)

            self._topics = {
                "image": image_topic,
                "job_done": job_done_topic,
                "coords": coords_topic,
                "ear": ear_topic,
                "mouth": mouth_topic,
                "safety": safety_topic,
            }
            self._spin_hz = max(10, int(spin_hz))

            self._executor: Optional[SingleThreadedExecutor] = None
            self._node: Optional[HmiRosNode] = None
            self._spin_timer: Optional[QTimer] = None

            self._last_image_emit = 0.0
            self._image_emit_min_dt = 1.0 / 30.0  # 최대 30fps로 UI emit 제한

        @pyqtSlot()
        def start(self):
            if not ROS_AVAILABLE:
                self.sig_ros_status.emit(False, "rclpy를 찾을 수 없습니다. (ROS2 환경에서 실행하세요)")
                return

            try:
                rclpy.init(args=None)

                def _log_info(text: str):
                    self.sig_log.emit("ROS", text)

                self._node = HmiRosNode(
                    image_topic=self._topics["image"],
                    job_done_topic=self._topics["job_done"],
                    coords_topic=self._topics["coords"],
                    ear_topic=self._topics["ear"],
                    mouth_topic=self._topics["mouth"],
                    safety_topic=self._topics["safety"],
                    on_image=self._on_image,
                    on_job_done=self._on_job_done,
                    on_coords=self._on_coords,
                    on_ear=self._on_ear,
                    on_mouth=self._on_mouth,
                    log_info=_log_info,
                )

                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self._node)

                interval_ms = max(1, int(1000 / self._spin_hz))
                self._spin_timer = QTimer(self)
                self._spin_timer.setInterval(interval_ms)
                self._spin_timer.timeout.connect(self._spin_once)
                self._spin_timer.start()

                self.sig_ros_status.emit(True, "ROS2 Connected")
                self.sig_log.emit("ROS", f"Spin loop started ({self._spin_hz} Hz)")

            except Exception as e:
                self.sig_ros_status.emit(False, f"ROS init 실패: {e}")
                self.sig_log.emit("ROS", traceback.format_exc())
                self._cleanup()

        @pyqtSlot()
        def stop(self):
            self.sig_log.emit("ROS", "Stopping ROS worker...")
            if self._spin_timer is not None:
                self._spin_timer.stop()
            self._cleanup()
            self.sig_ros_status.emit(False, "ROS2 Stopped")

        def _cleanup(self):
            try:
                if self._executor is not None and self._node is not None:
                    try:
                        self._executor.remove_node(self._node)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:
                pass

            try:
                if ROS_AVAILABLE and rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

            self._node = None
            self._executor = None
            self._spin_timer = None

        def _spin_once(self):
            if self._executor is None:
                return
            try:
                self._executor.spin_once(timeout_sec=0.0)
            except Exception as e:
                self.sig_ros_status.emit(False, f"Spin error: {e}")
                self.sig_log.emit("ROS", traceback.format_exc())
                self.stop()

        @pyqtSlot(bool)
        def publish_safety_recovery(self, value: bool = True):
            if self._node is None:
                self.sig_log.emit("ROS", "publish 실패: node가 없습니다")
                return
            try:
                self._node.publish_safety_recovery(value)
                self.sig_log.emit("UI", f"/safety/recovery publish: {bool(value)}")
            except Exception as e:
                self.sig_log.emit("ROS", f"/safety/recovery publish 실패: {e}")

        def _on_image(self, msg: RosImage):
            if self._node is None:
                return
            bridge = self._node.bridge
            if bridge is None:
                self.sig_log.emit("ROS", "cv_bridge가 없어 /eye/image_raw 변환 불가")
                return

            t = time.monotonic()
            if t - self._last_image_emit < self._image_emit_min_dt:
                return
            self._last_image_emit = t

            try:
                import cv2

                try:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    
                    # CUDA 사용 가능 여부 확인 후 처리
                    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(cv_img)  # CPU -> GPU 복사
                        gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB) # GPU 연산
                        rgb = gpu_rgb.download()  # GPU -> CPU 복사
                    else:
                        # CUDA 환경이 아닐 경우 CPU 연산으로 폴백(Fallback)
                        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        
                except Exception:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    rgb = cv_img

                h, w = rgb.shape[:2]
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    return

                bytes_per_line = 3 * w
                qimg = QImage(
                    rgb.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                ).copy()

                self.sig_image.emit(qimg)

            except Exception as e:
                self.sig_log.emit("ROS", f"이미지 변환 오류: {e}")

        def _on_job_done(self, msg: RosString):
            evt = parse_job_done(msg.data)
            if evt is None:
                return
            self.sig_job_done.emit(evt.job_id, evt.state, evt.raw)

        def _on_coords(self, msg: RosString):
            parsed = safe_json_loads(msg.data)
            self.sig_coords.emit(parsed if parsed is not None else msg.data)

        def _on_ear(self, msg: RosString):
            self.sig_log.emit("EAR", msg.data)

        def _on_mouth(self, msg: RosString):
            self.sig_log.emit("MOUTH", msg.data)


# =========================
# Demo Worker (no ROS)
# =========================

if QT_AVAILABLE:

    class DemoWorker(QObject):
        sig_ros_status = pyqtSignal(bool, str)
        sig_image = pyqtSignal(QImage)
        sig_job_done = pyqtSignal(int, str, object)
        sig_coords = pyqtSignal(object)
        sig_log = pyqtSignal(str, str)

        def __init__(self, parent: Optional[QObject] = None):
            super().__init__(parent)
            self._t_img: Optional[QTimer] = None
            self._t_misc: Optional[QTimer] = None
            self._job_id = 100
            self._frame = 0

        @pyqtSlot()
        def start(self):
            self.sig_ros_status.emit(True, "DEMO mode (no ROS)")
            self.sig_log.emit("DEMO", "Demo worker started")

            self._t_img = QTimer(self)
            self._t_img.setInterval(33)
            self._t_img.timeout.connect(self._emit_image)
            self._t_img.start()

            self._t_misc = QTimer(self)
            self._t_misc.setInterval(1000)
            self._t_misc.timeout.connect(self._emit_misc)
            self._t_misc.start()

        @pyqtSlot()
        def stop(self):
            if self._t_img:
                self._t_img.stop()
            if self._t_misc:
                self._t_misc.stop()
            self.sig_ros_status.emit(False, "DEMO stopped")

        @pyqtSlot(bool)
        def publish_safety_recovery(self, value: bool = True):
            self.sig_log.emit("UI", f"(DEMO) /safety/recovery publish: {bool(value)}")

        def _emit_image(self):
            w, h = 640, 480
            self._frame += 1
            buf = bytearray(w * h * 3)
            shift = self._frame % w
            for y in range(h):
                for x in range(w):
                    i = (y * w + x) * 3
                    r = (x + shift) % 256
                    g = (y * 2) % 256
                    b = (x + y + shift) % 256
                    buf[i] = r
                    buf[i + 1] = g
                    buf[i + 2] = b
            qimg = QImage(bytes(buf), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.sig_image.emit(qimg)

        def _emit_misc(self):
            self._job_id += 1

            coords = [
                {
                    "step": 1,
                    "action": "pick_screwdriver",
                    "label": "screwdriver",
                    "job_id": self._job_id,
                    "robot_x": None,
                    "robot_y": None,
                    "robot_z": None,
                    "depth_m": None,
                },
                {
                    "step": 2,
                    "action": "screw",
                    "label": "T1",
                    "job_id": self._job_id,
                    "robot_x": 123.45,
                    "robot_y": -67.89,
                    "robot_z": 201.5,
                    "depth_m": 0.215,
                },
            ]
            self.sig_coords.emit(coords)

            self.sig_log.emit("EAR", f"(DEMO) 작업 시작 {self._job_id}")
            self.sig_log.emit("MOUTH", f"(DEMO) 알겠습니다. Job {self._job_id} 수행")

            evt = {"job_id": self._job_id, "state": "done"}
            self.sig_job_done.emit(self._job_id, "done", evt)


# =========================
# Main Window
# =========================

if QT_AVAILABLE:

    class GeminiHMIWindow(QMainWindow):
        sig_publish_safety = pyqtSignal(bool)
        sig_stop_worker = pyqtSignal()

        def __init__(self, *, worker: QObject, is_demo: bool = False):
            super().__init__()

            self.setWindowTitle("Gemini Robotics ER - Integrated HMI")
            self.setMinimumSize(1366, 768)

            self._worker = worker
            self._is_demo = is_demo

            self._ts_ros_ok: bool = False
            self._ts_last_image: float = 0.0
            self._ts_last_coords: float = 0.0
            self._ts_last_ear: float = 0.0
            self._ts_last_mouth: float = 0.0

            self._build_ui()
            self._apply_hmi_style()
            self._wire_worker_signals()

            self._health_timer = QTimer(self)
            self._health_timer.setInterval(500)
            self._health_timer.timeout.connect(self._update_health)
            self._health_timer.start()

        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)

            grid = QGridLayout(root)
            grid.setContentsMargins(12, 12, 12, 12)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(12)

            self.gb_video = QGroupBox("LIVE CAMERA  /eye/image_raw")
            vlay = QVBoxLayout(self.gb_video)
            vlay.setContentsMargins(10, 12, 10, 10)
            vlay.setSpacing(8)

            self.video = ScaledImageLabel()
            self.video.setMinimumSize(720, 420)
            self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            vlay.addWidget(self.video)

            self.gb_status = QGroupBox("SYSTEM STATUS")
            slay = QVBoxLayout(self.gb_status)
            slay.setContentsMargins(10, 12, 10, 10)
            slay.setSpacing(10)

            self.led_ros = StatusLed("ROS2")
            self.led_video = StatusLed("Video")
            self.led_coords = StatusLed("/nerve/robot_coords")
            self.led_voice = StatusLed("Voice (/ear, /mouth)")

            slay.addWidget(self.led_ros)
            slay.addWidget(self.led_video)
            slay.addWidget(self.led_coords)
            slay.addWidget(self.led_voice)

            self.lbl_job = QLabel("JOB: -")
            self.lbl_job.setProperty("role", "status_label")

            self.lbl_state = QLabel("STATE: IDLE")
            self.lbl_state.setProperty("role", "status_label")

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("IDLE")

            slay.addWidget(self.lbl_job)
            slay.addWidget(self.lbl_state)
            slay.addWidget(self.progress)

            self.btn_safety = QPushButton("안전 복구\nSafety Recovery")
            self.btn_safety.setObjectName("SafetyButton")
            self.btn_safety.setMinimumHeight(92)
            self.btn_safety.clicked.connect(self._on_safety_clicked)
            slay.addWidget(self.btn_safety)

            slay.addItem(QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

            self.gb_coords = QGroupBox("ROBOT COORDS  /nerve/robot_coords")
            clay = QVBoxLayout(self.gb_coords)
            clay.setContentsMargins(10, 12, 10, 10)
            clay.setSpacing(8)

            self.lbl_target = QLabel("Target: -")
            self.lbl_target.setProperty("role", "status_label")
            clay.addWidget(self.lbl_target)

            self.tbl = QTableWidget(0, 7)
            self.tbl.setHorizontalHeaderLabels([
                "step",
                "action",
                "label",
                "x(mm)",
                "y(mm)",
                "z(mm)",
                "depth(m)",
            ])
            self.tbl.setAlternatingRowColors(True)
            self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.tbl.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.tbl.verticalHeader().setVisible(False)
            self.tbl.horizontalHeader().setStretchLastSection(True)
            clay.addWidget(self.tbl)

            self.gb_log = QGroupBox("SYSTEM LOG  (/ear, /mouth)")
            llay = QVBoxLayout(self.gb_log)
            llay.setContentsMargins(10, 12, 10, 10)
            llay.setSpacing(8)

            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setProperty("role", "log")
            self.log.setFont(QFont("Consolas", 10))
            llay.addWidget(self.log)

            btn_row = QHBoxLayout()
            self.btn_clear_log = QPushButton("Clear Log")
            self.btn_clear_log.clicked.connect(self.log.clear)
            btn_row.addWidget(self.btn_clear_log)
            btn_row.addStretch(1)
            llay.addLayout(btn_row)

            grid.addWidget(self.gb_video, 0, 0)
            grid.addWidget(self.gb_status, 0, 1)
            grid.addWidget(self.gb_coords, 1, 0)
            grid.addWidget(self.gb_log, 1, 1)

            grid.setRowStretch(0, 3)
            grid.setRowStretch(1, 2)
            grid.setColumnStretch(0, 3)
            grid.setColumnStretch(1, 1)

        def _apply_hmi_style(self):
            qss = """
            QWidget {
                background-color: #2B2B2B;
                color: #E0E0E0;
                font-family: "Segoe UI";
                font-size: 12px;
            }

            QGroupBox {
                background-color: #333333;
                border-top: 2px solid #5A5A5A;
                border-left: 2px solid #5A5A5A;
                border-right: 2px solid #1A1A1A;
                border-bottom: 2px solid #1A1A1A;
                border-radius: 8px;
                margin-top: 14px;
                padding: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #B0BEC5;
                font-weight: 600;
            }

            QLabel[role="status_label"] {
                color: #E0E0E0;
                font-weight: 600;
            }

            QLabel[role="video"] {
                background-color: #1F1F1F;
                border-top: 2px solid #5A5A5A;
                border-left: 2px solid #5A5A5A;
                border-right: 2px solid #111111;
                border-bottom: 2px solid #111111;
                border-radius: 6px;
                color: #777777;
                font-size: 18px;
                font-weight: 700;
            }

            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #4A4A4A, stop:1 #2F2F2F);
                border-top: 2px solid #6A6A6A;
                border-left: 2px solid #6A6A6A;
                border-right: 2px solid #151515;
                border-bottom: 2px solid #151515;
                border-radius: 8px;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #545454, stop:1 #343434);
            }

            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #2F2F2F, stop:1 #4A4A4A);
                border-top: 2px solid #151515;
                border-left: 2px solid #151515;
                border-right: 2px solid #6A6A6A;
                border-bottom: 2px solid #6A6A6A;
                padding-top: 10px;
                padding-left: 16px;
            }

            /* Safety Recovery - dominant orange gradient */
            QPushButton#SafetyButton {
                color: #FFFFFF;
                font-size: 16px;
                font-weight: 800;
                letter-spacing: 0.5px;
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #FF9800, stop:0.5 #EF6C00, stop:1 #D65A00);
                border-top: 3px solid #FFD180;
                border-left: 3px solid #FFD180;
                border-right: 3px solid #8C3A00;
                border-bottom: 3px solid #8C3A00;
                border-radius: 10px;
                padding: 16px 14px;
            }

            QPushButton#SafetyButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #D65A00, stop:0.5 #EF6C00, stop:1 #FF9800);
                border-top: 3px solid #8C3A00;
                border-left: 3px solid #8C3A00;
                border-right: 3px solid #FFD180;
                border-bottom: 3px solid #FFD180;
                padding-top: 18px;   /* 눌림(안으로 들어가는) 피드백 */
                padding-left: 16px;
            }

            QProgressBar {
                background-color: #1F1F1F;
                border-top: 2px solid #5A5A5A;
                border-left: 2px solid #5A5A5A;
                border-right: 2px solid #111111;
                border-bottom: 2px solid #111111;
                border-radius: 8px;
                text-align: center;
                height: 24px;
            }

            QProgressBar::chunk {
                background-color: #FFC107; /* 진행/대기 */
                border-radius: 8px;
            }

            QTableWidget {
                background-color: #262626;
                border-top: 2px solid #5A5A5A;
                border-left: 2px solid #5A5A5A;
                border-right: 2px solid #111111;
                border-bottom: 2px solid #111111;
                border-radius: 8px;
                gridline-color: #444444;
                selection-background-color: #2196F3; /* 통신 강조 */
                selection-color: #FFFFFF;
            }

            QHeaderView::section {
                background-color: #303030;
                color: #CFD8DC;
                padding: 6px;
                border: 1px solid #1A1A1A;
                font-weight: 700;
            }

            QTextEdit[role="log"] {
                background-color: #1F1F1F;
                border-top: 2px solid #5A5A5A;
                border-left: 2px solid #5A5A5A;
                border-right: 2px solid #111111;
                border-bottom: 2px solid #111111;
                border-radius: 8px;
            }
            """
            self.setStyleSheet(qss)

        def _wire_worker_signals(self):
            if hasattr(self._worker, "sig_ros_status"):
                self._worker.sig_ros_status.connect(self._on_ros_status)
            if hasattr(self._worker, "sig_image"):
                self._worker.sig_image.connect(self._on_image)
            if hasattr(self._worker, "sig_job_done"):
                self._worker.sig_job_done.connect(self._on_job_done)
            if hasattr(self._worker, "sig_coords"):
                self._worker.sig_coords.connect(self._on_coords)
            if hasattr(self._worker, "sig_log"):
                self._worker.sig_log.connect(self._on_log)

            self.sig_publish_safety.connect(self._worker.publish_safety_recovery)
            self.sig_stop_worker.connect(self._worker.stop)

        def _on_ros_status(self, ok: bool, message: str):
            self._ts_ros_ok = bool(ok)
            self._append_log("ROS", message)

        def _on_image(self, qimg: QImage):
            self._ts_last_image = time.monotonic()
            self.video.set_image(qimg)

        def _on_job_done(self, job_id: int, state: str, raw: object):
            st = (state or "").strip().lower()
            if job_id is not None and job_id >= 0:
                self.lbl_job.setText(f"JOB: {job_id}")
            self.lbl_state.setText(f"STATE: {state}")

            if st == "done":
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.progress.setFormat("DONE")
                self.progress.setStyleSheet(
                    "QProgressBar::chunk { background-color: #4CAF50; border-radius: 8px; }"
                )

        def _on_coords(self, payload: object):
            self._ts_last_coords = time.monotonic()

            if isinstance(payload, dict):
                payload = [payload]

            if isinstance(payload, list) and payload:
                self.progress.setRange(0, 0)  # running busy
                self.progress.setFormat("RUNNING")
                self.progress.setStyleSheet(
                    "QProgressBar::chunk { background-color: #FFC107; border-radius: 8px; }"
                )

                if all(isinstance(it, dict) for it in payload):
                    self._fill_coords_table(payload)
                    return

            self._append_log("NERVE", f"/nerve/robot_coords raw: {str(payload)[:200]}")

        def _on_log(self, source: str, text: str):
            t = time.monotonic()
            if source == "EAR":
                self._ts_last_ear = t
            elif source == "MOUTH":
                self._ts_last_mouth = t
            self._append_log(source, text)

        def _on_safety_clicked(self):
            self._append_log("UI", "Safety Recovery 버튼 클릭 → /safety/recovery 발행")
            self.sig_publish_safety.emit(True)

        def _append_log(self, source: str, text: str):
            color = "#CFD8DC"
            if source == "EAR":
                color = "#2196F3"
            elif source == "MOUTH":
                color = "#4CAF50"
            elif source in ("UI", "JOB"):
                color = "#FFC107"
            elif source == "ROS":
                color = "#B0BEC5"

            ts = now_ts_str()
            s = html.escape(source)
            msg = html.escape((text or "").strip())

            line = (
                f"<span style='color:#78909C'>[{ts}]</span> "
                f"<span style='color:{color}; font-weight:700'>[{s}]</span> "
                f"<span style='color:#E0E0E0'>{msg}</span>"
            )
            self.log.append(line)

        def _fill_coords_table(self, tasks: List[Dict[str, Any]]):
            tasks = tasks[:50]
            self.tbl.setRowCount(len(tasks))

            def _set(row: int, col: int, v: Any):
                self.tbl.setItem(row, col, QTableWidgetItem("" if v is None else str(v)))

            for r, t in enumerate(tasks):
                _set(r, 0, t.get("step"))
                _set(r, 1, t.get("action"))
                _set(r, 2, t.get("label"))
                _set(r, 3, f"{_as_float(t.get('robot_x')):.2f}" if _as_float(t.get("robot_x")) is not None else "-")
                _set(r, 4, f"{_as_float(t.get('robot_y')):.2f}" if _as_float(t.get("robot_y")) is not None else "-")
                _set(r, 5, f"{_as_float(t.get('robot_z')):.2f}" if _as_float(t.get("robot_z")) is not None else "-")
                _set(r, 6, f"{_as_float(t.get('depth_m')):.3f}" if _as_float(t.get("depth_m")) is not None else "-")

            self.tbl.resizeColumnsToContents()

        def _update_health(self):
            now = time.monotonic()

            self.led_ros.set_state(self._ts_ros_ok, "#4CAF50" if self._ts_ros_ok else "#666666")

            video_ok = (now - self._ts_last_image) < 1.0
            self.led_video.set_state(video_ok, "#2196F3" if video_ok else "#666666")
            if not video_ok and self._ts_last_image > 0:
                self.video.clear_image("NO SIGNAL")

            coords_ok = (now - self._ts_last_coords) < 2.0
            self.led_coords.set_state(coords_ok, "#2196F3" if coords_ok else "#666666")

            voice_ok = ((now - self._ts_last_ear) < 3.0) or ((now - self._ts_last_mouth) < 3.0)
            self.led_voice.set_state(voice_ok, "#2196F3" if voice_ok else "#666666")

        def closeEvent(self, e):
            try:
                self.sig_stop_worker.emit()
            except Exception:
                pass
            super().closeEvent(e)


# =========================
# Self-test (no Qt / no ROS)
# =========================

def self_test() -> int:
    print("[selftest] safe_json_loads / parse_job_done")
    samples = [
        '{"job_id": 12, "state": "done"}',
        '{"job_id": "13", "state": "DONE"}',
        'not-json',
        '',
        '   ',
    ]
    for s in samples:
        evt = parse_job_done(s)
        print(f"  input={s!r}")
        print(f"  parsed={evt}")

    coords_samples = [
        '[{"step":1,"action":"screw","label":"T1","robot_x":1.2,"robot_y":3.4,"robot_z":200.1,"depth_m":0.22}]',
        '[123.4, -55.6, 201.5]',
        '{"robot_x": 1, "robot_y":2}',
        'oops',
    ]
    print("\n[selftest] /nerve/robot_coords parsing")
    for s in coords_samples:
        parsed = safe_json_loads(s)
        print(f"  input={s!r}")
        print(f"  parsed_type={type(parsed).__name__}, parsed={parsed}")

    print("\n[selftest] OK")
    return 0


# =========================
# Entry
# =========================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run without ROS (synthetic data)")
    ap.add_argument("--selftest", action="store_true", help="Run parser self-test only (no GUI)")

    ap.add_argument("--spin-hz", type=int, default=100, help="ROS spin_once rate (Hz)")

    ap.add_argument("--image-topic", default="/eye/image_raw")
    ap.add_argument("--job-done-topic", default="/muscle/job_done")
    ap.add_argument("--coords-topic", default="/nerve/robot_coords")
    ap.add_argument("--ear-topic", default="/ear/speech_text")
    ap.add_argument("--mouth-topic", default="/mouth/speech_text")
    ap.add_argument("--safety-topic", default="/safety/recovery")

    args = ap.parse_args()

    if args.selftest:
        return self_test()

    if not QT_AVAILABLE:
        print("[ERROR] PyQt6가 설치되어 있지 않습니다.")
        print("  Ubuntu 22.04(ROS2 Humble) 예시:")
        print("    sudo apt install -y python3-pyqt6")
        return 2

    app = QApplication(sys.argv)

    thread = QThread()

    if args.demo:
        worker = DemoWorker()
    else:
        worker = RosWorker(
            image_topic=args.image_topic,
            job_done_topic=args.job_done_topic,
            coords_topic=args.coords_topic,
            ear_topic=args.ear_topic,
            mouth_topic=args.mouth_topic,
            safety_topic=args.safety_topic,
            spin_hz=args.spin_hz,
        )

    worker.moveToThread(thread)
    thread.started.connect(worker.start)
    thread.finished.connect(worker.deleteLater)

    win = GeminiHMIWindow(worker=worker, is_demo=args.demo)
    win.show()

    def _on_about_to_quit():
        try:
            win.sig_stop_worker.emit()
        except Exception:
            pass
        thread.quit()
        thread.wait(1500)

    app.aboutToQuit.connect(_on_about_to_quit)

    thread.start()
    rc = app.exec()

    try:
        thread.quit()
        thread.wait(1500)
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    raise SystemExit(main())