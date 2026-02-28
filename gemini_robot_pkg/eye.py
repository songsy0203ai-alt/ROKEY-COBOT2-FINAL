#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eye_pub_images.py
- 기존 eye.py 기능 유지 + ✅ UI용 영상 토픽 추가 발행

기존 Pub(유지):
  /eye/terminal_centers     (std_msgs/String)  # timer/relay/lamp/switch/power 감지 중심 [ny,nx]
  /eye/driver_rs_center     (std_msgs/String)  # RealSense driver best {"found":bool,"point":[ny,nx],...}
  /eye/driver_web_center    (std_msgs/String)  # Webcam driver best ...
  /eye/aligned_depth        (sensor_msgs/Image) # RealSense depth 16UC1
  /eye/depth_info           (std_msgs/String)   # intrinsics + depth_scale

✅ 추가 Pub:
  /eye/image_main           (sensor_msgs/Image) # RealSense color bgr8 (UI용)
  /eye/image_cam9           (sensor_msgs/Image) # Webcam(cam9) bgr8 (UI용)

Env:
  EYE_WIDTH=640
  EYE_HEIGHT=480
  EYE_FPS=30
  EYE_CAM9_INDEX=8
  EYE_RS_SERIAL= (optional)
  EYE_PUB_IMAGES=1/0   # default 1
  EYE_PUB_FPS=10       # 이미지 토픽 발행 최대 fps
  EYE_SHOW_WINDOWS=0/1 # default 0 (GUI 없이 서버에서 돌릴 때 0 추천)

Model paths (same defaults as your eye.py):
  EYE_TIMER_MODEL, EYE_RELAY_MODEL, EYE_LAMP_MODEL, EYE_SWITCH_MODEL, EYE_POWER_MODEL, EYE_DRIVER_MODEL

Dependencies:
  pip install ultralytics opencv-python pyrealsense2 cv_bridge

주의:
- UI(workflow_ui_cuda_korean.py / eye_ui.py)는 /eye/image_main /eye/image_cam9 를 구독합니다.
  기존 eye.py에는 이 토픽이 없어서 UI 영상이 안 뜰 수 있어, 이 버전을 사용하세요.
"""

import os
import json
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_RS = True
except Exception:
    HAS_RS = False


def _env(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return v if v else default


def _to_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


class IntegratedEyeNode(Node):
    def __init__(self):
        super().__init__("eye_node")

        # ===== 모델 경로 설정 =====
        timer_model_path  = _env("EYE_TIMER_MODEL",  "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/best_timer_v_260225.pt")
        relay_model_path  = _env("EYE_RELAY_MODEL",  "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/best_relay.pt")
        lamp_model_path   = _env("EYE_LAMP_MODEL",   "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/best_lamp_v_260225.pt")
        switch_model_path = _env("EYE_SWITCH_MODEL", "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/best_switch.pt")
        power_model_path  = _env("EYE_POWER_MODEL",  "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/best_power.pt")

        default_driver_model = "/home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/trained_models/webcam_driver_v4.pt"
        driver_model_path = _env("EYE_DRIVER_MODEL", default_driver_model)

        # ===== YOLO 로드 (CUDA) =====
        self.timer_model  = YOLO(timer_model_path).to("cuda")
        self.relay_model  = YOLO(relay_model_path).to("cuda")
        self.lamp_model   = YOLO(lamp_model_path).to("cuda")
        self.switch_model = YOLO(switch_model_path).to("cuda")
        self.power_model  = YOLO(power_model_path).to("cuda")
        self.driver_model = YOLO(driver_model_path).to("cuda")

        # ===== ROS Pub =====
        self.coord_pub = self.create_publisher(String, "/eye/terminal_centers", 10)
        self.driver_rs_pub = self.create_publisher(String, "/eye/driver_rs_center", 10)
        self.driver_web_pub = self.create_publisher(String, "/eye/driver_web_center", 10)

        self.depth_pub = self.create_publisher(Image, "/eye/aligned_depth", 2)
        self.info_pub  = self.create_publisher(String, "/eye/depth_info", 10)

        # ✅ UI 영상용
        self.pub_images = _to_bool(_env("EYE_PUB_IMAGES", "1"), True)
        self.pub_fps = float(_env("EYE_PUB_FPS", "10"))
        self._last_pub_t_main = 0.0
        self._last_pub_t_cam9 = 0.0
        self.bridge = CvBridge()
        self.img_main_pub = self.create_publisher(Image, "/eye/image_main", 2)
        self.img_cam9_pub = self.create_publisher(Image, "/eye/image_cam9", 2)

        # ===== 카메라 설정 =====
        self.img_w = int(_env("EYE_WIDTH", "640"))
        self.img_h = int(_env("EYE_HEIGHT", "480"))
        self.fps   = int(_env("EYE_FPS", "30"))

        self.show_windows = _to_bool(_env("EYE_SHOW_WINDOWS", "0"), False)

        # RealSense
        self.use_rs = False
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intr_sent = False

        if HAS_RS:
            self.get_logger().info("✅ pyrealsense2 감지: RealSense(color+depth) 모드로 실행합니다.")
            try:
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                serial = _env("EYE_RS_SERIAL", "")
                if serial:
                    cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, self.img_w, self.img_h, rs.format.bgr8, self.fps)
                cfg.enable_stream(rs.stream.depth, self.img_w, self.img_h, rs.format.z16, self.fps)
                profile = self.pipeline.start(cfg)
                self.align = rs.align(rs.stream.color)

                dev = profile.get_device()
                depth_sensor = dev.first_depth_sensor()
                self.depth_scale = float(depth_sensor.get_depth_scale())
                self.use_rs = True
            except Exception as e:
                self.get_logger().warn(f"⚠️ RealSense start 실패 → VideoCapture 폴백. 원인: {e}")
                self.use_rs = False

        if not self.use_rs:
            cam_index = int(_env("EYE_CAM_INDEX", "0"))
            self.cap = cv2.VideoCapture(cam_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)
            self.get_logger().warn(f"⚠️ VideoCapture(main) 모드 실행. cam_index={cam_index}")
        else:
            self.cap = None

        # Webcam(cap9)
        self.cam9_index = int(_env("EYE_CAM9_INDEX", "8"))
        self.cap9 = cv2.VideoCapture(self.cam9_index)
        if not self.cap9.isOpened():
            self.get_logger().error(f"❌ cam9 인덱스 {self.cam9_index} 열기 실패")
        else:
            self.cap9.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_w)
            self.cap9.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)
            self.get_logger().info(f"✅ cam9 연결 성공 (index={self.cam9_index})")

        if self.show_windows:
            self.window_main = "Eye(Main)"
            self.window_cam9 = "Eye(Cam9)"
            cv2.namedWindow(self.window_main, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_main, 1280, 720)
            cv2.namedWindow(self.window_cam9, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_cam9, 640, 480)

        self.timer = self.create_timer(0.1, self.inference_callback)
        self.get_logger().info("✅ Eye node ready (driver publish + image topics).")

    # -------------------------
    # Depth info publish (once)
    # -------------------------
    def _publish_depth_info_once(self, color_frame):
        if self.intr_sent:
            return
        try:
            vp = color_frame.profile.as_video_stream_profile()
            intr = vp.get_intrinsics()
            info = {
                "width": int(intr.width), "height": int(intr.height),
                "fx": float(intr.fx), "fy": float(intr.fy),
                "ppx": float(intr.ppx), "ppy": float(intr.ppy),
                "depth_scale": float(self.depth_scale if self.depth_scale is not None else 0.0),
            }
            self.info_pub.publish(String(data=json.dumps(info, ensure_ascii=False)))
            self.intr_sent = True
        except Exception as e:
            self.get_logger().warn(f"⚠️ depth_info 발행 실패: {e}")

    def _publish_depth_image(self, depth_raw: np.ndarray):
        try:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "realsense_color"
            msg.height, msg.width = depth_raw.shape
            msg.encoding = "16UC1"
            msg.step = msg.width * 2
            msg.data = depth_raw.tobytes()
            self.depth_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"⚠️ depth 발행 실패: {e}")

    def _publish_bgr_image(self, pub, frame_bgr: np.ndarray, frame_id: str, last_pub_t_attr: str):
        if not self.pub_images:
            return
        now = time.time()
        last_t = getattr(self, last_pub_t_attr, 0.0)
        if self.pub_fps > 0 and (now - last_t) < (1.0 / self.pub_fps):
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            pub.publish(msg)
            setattr(self, last_pub_t_attr, now)
        except Exception as e:
            self.get_logger().warn(f"⚠️ image publish 실패({frame_id}): {e}")

    # -------------------------
    # Best detection (driver)
    # -------------------------
    def _detect_best_point(self, model: YOLO, frame, img_w: int, img_h: int, conf: float = 0.15):
        """
        return: (label, score, [ny,nx]) or None
        """
        try:
            results = model(frame, verbose=False, conf=conf)
        except Exception:
            return None

        best = None  # (score, label, ny, nx)
        if not results:
            return None

        for r in results:
            # OBB
            if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                for obb in r.obb:
                    try:
                        class_id = int(obb.cls[0])
                        label = model.names[class_id]
                        score = float(obb.conf[0])

                        center = obb.xywhr[0].cpu().numpy()
                        cx, cy = float(center[0]), float(center[1])

                        ny = int((cy / float(img_h)) * 1000.0)
                        nx = int((cx / float(img_w)) * 1000.0)

                        if best is None or score > best[0]:
                            best = (score, label, ny, nx)
                    except Exception:
                        continue

            # BBox
            elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    try:
                        class_id = int(box.cls[0])
                        label = model.names[class_id]
                        score = float(box.conf[0])

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

                        ny = int((cy / float(img_h)) * 1000.0)
                        nx = int((cx / float(img_w)) * 1000.0)

                        if best is None or score > best[0]:
                            best = (score, label, ny, nx)
                    except Exception:
                        continue

        if best is None:
            return None
        score, label, ny, nx = best
        return label, float(score), [int(ny), int(nx)]

    # -------------------------
    # Draw + list helper
    # -------------------------
    def _run_model_draw(self, model: YOLO, frame, color, out_list, img_w, img_h, conf=0.15):
        try:
            results = model(frame, verbose=False, conf=conf)
        except Exception:
            return
        if not results:
            return

        for r in results:
            # OBB
            if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                for obb in r.obb:
                    try:
                        class_id = int(obb.cls[0])
                        label = model.names[class_id]
                        score = float(obb.conf[0])

                        points = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                        center_coords = obb.xywhr[0].cpu().numpy()
                        cx, cy = float(center_coords[0]), float(center_coords[1])

                        norm_y = int((cy / img_h) * 1000)
                        norm_x = int((cx / img_w) * 1000)
                        out_list.append({"label": label, "point": [norm_y, norm_x]})

                        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                        cv2.putText(frame, f"{label} {score:.2f}",
                                    (int(points[0][0]), max(15, int(points[0][1]) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception:
                        continue

            # BBox
            elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    try:
                        class_id = int(box.cls[0])
                        label = model.names[class_id]
                        score = float(box.conf[0])

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

                        norm_y = int((cy / img_h) * 1000)
                        norm_x = int((cx / img_w) * 1000)
                        out_list.append({"label": label, "point": [norm_y, norm_x]})

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{label} {score:.2f}",
                                    (int(x1), max(15, int(y1) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception:
                        continue

    # -------------------------
    # Main inference loop
    # -------------------------
    def inference_callback(self):
        # 1) Main frame (RealSense or VideoCapture)
        frame_main = None
        if self.use_rs and self.pipeline is not None and self.align is not None:
            try:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if color and depth:
                    self._publish_depth_info_once(color)
                    frame_main = np.asanyarray(color.get_data())
                    depth_raw = np.asanyarray(depth.get_data())
                    self._publish_depth_image(depth_raw)
            except Exception:
                frame_main = None
        else:
            ret, frame_main = self.cap.read() if self.cap is not None else (False, None)
            if not ret:
                frame_main = None

        # 2) Main inference + pubs
        if frame_main is not None:
            h, w, _ = frame_main.shape
            detections = []
            self._run_model_draw(self.timer_model,  frame_main, (0, 255, 0),   detections, w, h)
            self._run_model_draw(self.relay_model,  frame_main, (255, 0, 0),   detections, w, h)
            self._run_model_draw(self.lamp_model,   frame_main, (0, 255, 255), detections, w, h)
            self._run_model_draw(self.switch_model, frame_main, (255, 255, 0), detections, w, h)
            self._run_model_draw(self.power_model,  frame_main, (255, 255, 0), detections, w, h)

            self.coord_pub.publish(String(data=json.dumps(detections, ensure_ascii=False)))

            # driver(best) for main
            best = self._detect_best_point(self.driver_model, frame_main, w, h, conf=0.15)
            if best is None:
                payload = {"found": False, "t": time.time()}
            else:
                label, score, pt = best
                payload = {"found": True, "label": label, "score": float(score), "point": pt, "t": time.time()}
            self.driver_rs_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

            # ✅ image_main publish
            self._publish_bgr_image(self.img_main_pub, frame_main, "eye_main_bgr8", "_last_pub_t_main")

            if self.show_windows:
                cv2.imshow(self.window_main, frame_main)

        # 3) Webcam cam9: driver(best) + publish image
        if self.cap9 is not None and self.cap9.isOpened():
            ret9, frame9 = self.cap9.read()
            if ret9 and frame9 is not None:
                h9, w9, _ = frame9.shape

                # optional draw
                _tmp = []
                self._run_model_draw(self.driver_model, frame9, (0, 0, 255), _tmp, w9, h9)

                best9 = self._detect_best_point(self.driver_model, frame9, w9, h9, conf=0.15)
                if best9 is None:
                    payload9 = {"found": False, "t": time.time()}
                else:
                    label9, score9, pt9 = best9
                    payload9 = {"found": True, "label": label9, "score": float(score9), "point": pt9, "t": time.time()}
                self.driver_web_pub.publish(String(data=json.dumps(payload9, ensure_ascii=False)))

                # ✅ image_cam9 publish
                self._publish_bgr_image(self.img_cam9_pub, frame9, "eye_cam9_bgr8", "_last_pub_t_cam9")

                if self.show_windows:
                    cv2.imshow(self.window_cam9, frame9)

        if self.show_windows:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().warn("q pressed -> shutdown")
                self.destroy_node()
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedEyeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if getattr(node, "cap", None):
                node.cap.release()
        except Exception:
            pass
        try:
            if getattr(node, "cap9", None):
                node.cap9.release()
        except Exception:
            pass
        try:
            if getattr(node, "use_rs", False) and getattr(node, "pipeline", None):
                node.pipeline.stop()
        except Exception:
            pass
        if getattr(node, "show_windows", False):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
