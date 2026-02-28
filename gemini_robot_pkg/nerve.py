#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nerve_passthrough_extras.py
- 기존 nerve_fixed_qos.py 기반
- 변경점: Brain task dict의 추가 필드(wait_sec 등)를 "그대로" 유지하면서
          robot_x/robot_y/robot_z/depth_m만 덮어써서 /nerve/robot_coords로 전달.

즉, Brain → Nerve → Muscle 파이프라인에서
  - wait_sec, 기타 커스텀 키가 유실되지 않게 함.

입력
- /brain/normalized_coords (std_msgs/String, JSON list or dict)

추가 입력(Z용)
- /eye/aligned_depth (sensor_msgs/Image, 16UC1)
- /eye/depth_info   (std_msgs/String, JSON: fx,fy,ppx,ppy,width,height,depth_scale)

추가 입력(연결)
- /muscle/current_pose (std_msgs/String): "x,y,z,rx,ry,rz" (mm,deg)

출력
- /nerve/robot_coords (std_msgs/String, JSON list)
  - task 원본 키 유지 + robot_x, robot_y, robot_z, depth_m 추가/갱신
"""

import os
import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def pose6_to_T_mm(x, y, z, rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(float(rx_deg))
    ry = np.deg2rad(float(ry_deg))
    rz = np.deg2rad(float(rz_deg))
    R = _rotz(rz) @ _roty(ry) @ _rotx(rx)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [float(x), float(y), float(z)]
    return T


class GeminiNerve:
    def __init__(self, calibration_path: str):
        self.calibration_path = calibration_path
        self.homography_matrix = None
        self.load_calibration()

    def load_calibration(self) -> bool:
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, 'r') as f:
                    data = json.load(f)
                self.homography_matrix = np.array(data['matrix'], dtype=float)
                return True
            except Exception:
                self.homography_matrix = None
                return False
        return False

    @staticmethod
    def denormalize_pixel(y_norm, x_norm, img_w, img_h):
        px = (float(x_norm) / 1000.0) * float(img_w)
        py = (float(y_norm) / 1000.0) * float(img_h)
        return px, py

    def convert_to_robot_coords(self, y_norm, x_norm, img_w=640, img_h=480):
        if self.homography_matrix is None:
            return None
        px, py = self.denormalize_pixel(y_norm, x_norm, img_w, img_h)
        p = np.array([[px], [py], [1.0]], dtype=float)
        rp = np.dot(self.homography_matrix, p)
        w = float(rp[2][0])
        if abs(w) < 1e-9:
            return None
        rx = float(rp[0][0]) / w
        ry = float(rp[1][0]) / w
        return [rx, ry]


class DepthHelper:
    def __init__(self):
        self.depth_img = None
        self.w = 0
        self.h = 0
        self.depth_scale = None
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None

    def update_info(self, info: dict):
        self.w = int(info.get("width", self.w or 0))
        self.h = int(info.get("height", self.h or 0))
        if info.get("fx") is not None:
            self.fx = float(info["fx"])
        if info.get("fy") is not None:
            self.fy = float(info["fy"])
        if info.get("ppx") is not None:
            self.ppx = float(info["ppx"])
        if info.get("ppy") is not None:
            self.ppy = float(info["ppy"])
        if info.get("depth_scale") is not None:
            self.depth_scale = float(info["depth_scale"])

    def update_depth_image(self, msg: Image):
        if msg.height <= 0 or msg.width <= 0:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            self.depth_img = arr
            self.h = int(msg.height)
            self.w = int(msg.width)
        except Exception:
            self.depth_img = None

    def depth_median_m(self, u: int, v: int, r: int = 5):
        if self.depth_img is None or self.depth_scale is None:
            return None
        if u < 0 or v < 0 or u >= self.w or v >= self.h:
            return None

        u0 = max(0, u - r)
        u1 = min(self.w - 1, u + r)
        v0 = max(0, v - r)
        v1 = min(self.h - 1, v + r)

        roi = self.depth_img[v0:v1 + 1, u0:u1 + 1].astype(np.float32)
        roi = roi[roi > 0]
        if roi.size == 0:
            return None

        med_raw = float(np.median(roi))
        return med_raw * float(self.depth_scale)

    def deproject_cam_mm(self, u: int, v: int, depth_m: float):
        if self.fx is None or self.fy is None or self.ppx is None or self.ppy is None:
            return None
        if depth_m is None:
            return None
        Z = float(depth_m)
        X = (float(u) - float(self.ppx)) / float(self.fx) * Z
        Y = (float(v) - float(self.ppy)) / float(self.fy) * Z
        return np.array([X * 1000.0, Y * 1000.0, Z * 1000.0, 1.0], dtype=float)


class NerveNode(Node):
    def __init__(self):
        super().__init__('nerve_node')

        # QoS: /nerve/robot_coords를 Transient Local(라치)로 발행(기본 ON)
        self.enable_latched_qos = os.getenv('NERVE_LATCHED_QOS', '1').strip().lower() not in ('0','false','no')
        if self.enable_latched_qos:
            self.qos_latched = QoSProfile(depth=1)
            self.qos_latched.history = HistoryPolicy.KEEP_LAST
            self.qos_latched.reliability = ReliabilityPolicy.RELIABLE
            self.qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        else:
            self.qos_latched = None

        calib_path = os.path.expanduser(os.getenv(
            "NERVE_CALIB_PATH",
            "~/cobot_ws/src/cobot2_ws/gemini_robot_pkg/data/calibration/matrix.json",
        ))
        self.nerve = GeminiNerve(calib_path)
        if self.nerve.homography_matrix is None:
            self.get_logger().warn(f"⚠️ 캘리브레이션 파일이 없습니다: {calib_path}")

        handeye_path = os.path.expanduser(os.getenv(
            "NERVE_HAND_EYE_PATH",
            "~/cobot_ws/src/cobot2_ws/gemini_robot_pkg/data/calibration/T_gripper2camera.npy",
        ))
        self.T_gripper_cam = None
        if os.path.exists(handeye_path):
            try:
                self.T_gripper_cam = np.load(handeye_path).astype(float)
                if self.T_gripper_cam.shape != (4, 4):
                    self.get_logger().warn(f"⚠️ hand-eye npy가 4x4가 아닙니다: {handeye_path}")
                    self.T_gripper_cam = None
                else:
                    self.get_logger().info(f"✅ hand-eye npy 로드 성공: {handeye_path}")
            except Exception as e:
                self.get_logger().warn(f"⚠️ hand-eye npy 로드 실패: {e} ({handeye_path})")
        else:
            self.get_logger().warn(f"⚠️ hand-eye npy 파일이 없습니다: {handeye_path}")

        # 기본 스캔 포즈(폴백)
        scan_pose = os.getenv("NERVE_SCAN_POSE", "423.92,-147.13,402.39,168.05,-179.78,167.78")
        parts = [p.strip() for p in scan_pose.split(",")]
        if len(parts) != 6:
            parts = ["423.92", "-147.13", "402.39", "168.05", "-179.78", "167.78"]
        sx, sy, sz, srx, sry, srz = map(float, parts)
        self.T_base_tcp_scan = pose6_to_T_mm(sx, sy, sz, srx, sry, srz)

        # ✅ 현재 TCP 포즈(연결): /muscle/current_pose 수신 시 갱신
        self.T_base_tcp_current = None  # type: ignore

        self.expected_z_mm = float(os.getenv("NERVE_EXPECTED_Z_MM", "201.5"))
        self.z_offset_mm = float(os.getenv("NERVE_Z_OFFSET_MM", "0.0"))

        self.img_w = int(os.getenv("NERVE_IMG_W", "640"))
        self.img_h = int(os.getenv("NERVE_IMG_H", "480"))

        self.use_depth_info_dim = str(os.getenv("NERVE_USE_DEPTH_INFO_DIM", "0")).strip().lower() in ("1", "true", "yes", "y", "on")

        self.depth = DepthHelper()

        # ✅ 최근 Brain task 캐시 (pose 업데이트 시 Z 재계산을 위해)
        self._last_brain_tasks = None  # type: ignore

        self.sub = self.create_subscription(String, "/brain/normalized_coords", self._cb, 10)
        if self.enable_latched_qos and self.qos_latched is not None:
            self.pub = self.create_publisher(String, "/nerve/robot_coords", self.qos_latched)
        else:
            self.pub = self.create_publisher(String, "/nerve/robot_coords", 10)

        self.depth_sub = self.create_subscription(Image, "/eye/aligned_depth", self._depth_cb, 2)
        self.info_sub = self.create_subscription(String, "/eye/depth_info", self._info_cb, 10)

        # ✅ 연결: Muscle의 pose 피드백 구독
        self.pose_sub = self.create_subscription(String, "/muscle/current_pose", self._pose_cb, 10)

        self.get_logger().info(
            "✅ Nerve node ready. Waiting /brain/normalized_coords (+ optional /eye depth topics, /muscle/current_pose)\n"
            f"   QoS publish(/nerve/robot_coords)={'TRANSIENT_LOCAL' if self.enable_latched_qos else 'VOLATILE'}"
        )

    def _depth_cb(self, msg: Image):
        self.depth.update_depth_image(msg)

    def _info_cb(self, msg: String):
        try:
            info = json.loads(msg.data)
            if isinstance(info, dict):
                self.depth.update_info(info)
                if self.use_depth_info_dim:
                    if info.get("width") is not None:
                        self.img_w = int(info["width"])
                    if info.get("height") is not None:
                        self.img_h = int(info["height"])
        except Exception:
            pass

    def _pose_cb(self, msg: String):
        s = (msg.data or "").strip()
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 6:
            return
        try:
            x, y, z, rx, ry, rz = map(float, parts)
        except Exception:
            return

        self.T_base_tcp_current = pose6_to_T_mm(x, y, z, rx, ry, rz)

        if self._last_brain_tasks:
            try:
                out = self._convert_tasks(self._last_brain_tasks)
                self.pub.publish(String(data=json.dumps(out, ensure_ascii=False)))
            except Exception as e:
                self.get_logger().warn(f"⚠️ pose 기반 재발행 실패: {e}")

    def _get_T_base_tcp(self):
        return self.T_base_tcp_current if self.T_base_tcp_current is not None else self.T_base_tcp_scan

    def _compute_robot_z(self, px: float, py: float):
        if self.T_gripper_cam is None:
            return None, None

        u = int(round(px))
        v = int(round(py))
        depth_m = self.depth.depth_median_m(u, v, r=5)
        if depth_m is None:
            return None, None

        p_cam = self.depth.deproject_cam_mm(u, v, depth_m)
        if p_cam is None:
            return depth_m, None

        T_base_tcp = self._get_T_base_tcp()

        zA = None
        try:
            T_inv = np.linalg.inv(self.T_gripper_cam)
            p_base_A = (T_base_tcp @ T_inv @ p_cam)
            zA = float(p_base_A[2]) + self.z_offset_mm
        except Exception:
            zA = None

        zB = None
        try:
            p_base_B = (T_base_tcp @ self.T_gripper_cam @ p_cam)
            zB = float(p_base_B[2]) + self.z_offset_mm
        except Exception:
            zB = None

        cand = []
        if zA is not None and np.isfinite(zA):
            cand.append(("A", zA, abs(zA - self.expected_z_mm)))
        if zB is not None and np.isfinite(zB):
            cand.append(("B", zB, abs(zB - self.expected_z_mm)))

        if not cand:
            return depth_m, None

        cand.sort(key=lambda x: x[2])
        return depth_m, float(cand[0][1])

    def _convert_tasks(self, tasks: list):
        if isinstance(tasks, dict):
            tasks = [tasks]
        if not isinstance(tasks, list) or not tasks:
            return []

        any_execute = any(_to_bool(t.get("execute", False)) for t in tasks)
        try:
            main_job_id = int(tasks[0].get("job_id", 0))
        except Exception:
            main_job_id = 0

        out = []
        for t in tasks:
            # ✅ 원본 task의 모든 키를 유지(복사)
            base = dict(t) if isinstance(t, dict) else {}

            label = str(base.get("label", "")).strip()
            terminal = str(base.get("terminal", label)).strip()
            base["terminal"] = terminal

            pt = base.get("point", None)

            robot_xy = None
            robot_z = None
            depth_m = None

            if (pt is not None) and isinstance(pt, list) and len(pt) == 2:
                y_norm, x_norm = pt

                if self.nerve.homography_matrix is not None:
                    robot_xy = self.nerve.convert_to_robot_coords(y_norm, x_norm, img_w=self.img_w, img_h=self.img_h)

                px, py = self.nerve.denormalize_pixel(y_norm, x_norm, self.img_w, self.img_h)
                depth_m, robot_z = self._compute_robot_z(px, py)

            # ✅ 덮어쓰기/추가
            base["execute"] = any_execute
            base["job_id"] = main_job_id
            base["robot_x"] = (robot_xy[0] if robot_xy else base.get("robot_x"))
            base["robot_y"] = (robot_xy[1] if robot_xy else base.get("robot_y"))
            base["depth_m"] = depth_m if depth_m is not None else base.get("depth_m")
            base["robot_z"] = robot_z if robot_z is not None else base.get("robot_z")

            out.append(base)

        return out

    def _cb(self, msg: String):
        try:
            tasks = json.loads(msg.data)
            if isinstance(tasks, dict):
                tasks = [tasks]
            if not isinstance(tasks, list) or not tasks:
                return

            self._last_brain_tasks = tasks

            out = self._convert_tasks(tasks)
            self.pub.publish(String(data=json.dumps(out, ensure_ascii=False)))

        except Exception as e:
            self.get_logger().error(f"❌ Nerve parse/convert error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NerveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
