# /home/rokey/cobot_ws/src/cobot2_ws/gemini_robot_pkg/gemini_robot_pkg/ear.py

"""
ear.py
- 역할: 주변 음성을 STT로 텍스트 변환 후 발행
- Pub: /ear/speech_text (std_msgs/String)  # raw text
"""

import os
import time
import tempfile

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from google import genai
from google.genai import types


class EarNode(Node):
    def __init__(self):
        super().__init__('ear_node')

        # Debug (도메인/미들웨어 확인)
        self.get_logger().info(
            f"🌐 ROS_DOMAIN_ID={os.getenv('ROS_DOMAIN_ID','(unset)')}, "
            f"RMW_IMPLEMENTATION={os.getenv('RMW_IMPLEMENTATION','(unset)')}"
        )

        # ✅ 환경변수로 키 받기
        self.api_key = os.getenv("GEMINI_API_KEY_EAR", "").strip()
        if not self.api_key:
            self.get_logger().error(
                "❌ GEMINI_API_KEY_EAR 환경변수가 없습니다.\n"
                "   예) export GEMINI_API_KEY_EAR='AIza...'"
            )
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

        self.model_id = os.getenv("EAR_MODEL_ID", "gemini-2.5-flash")
        self.duration = float(os.getenv("EAR_DURATION_SEC", "5.0"))
        self.samplerate = int(os.getenv("EAR_SAMPLE_RATE", "16000"))
        self.threshold = float(os.getenv("EAR_THRESHOLD", "0.07"))

        # ✅ 너무 긴 cooldown 때문에 다음 명령이 안 들어오는 문제가 자주 발생해서 기본값을 2초로 줄였습니다.
        # (필요하면 EAR_COOLDOWN_SEC 환경변수로 조절)
        self.cooldown_sec = float(os.getenv("EAR_COOLDOWN_SEC", "2.0"))
        self._cooldown_until = 0.0

        self.is_processing = False
        self.pub = self.create_publisher(String, '/ear/speech_text', 10)

        self.create_timer(1.0, self._tick)
        self.get_logger().info(f"👂 Ear node ready (threshold={self.threshold}, duration={self.duration}s)")

    def _tick(self):
        # cooldown 중이면 다음 tick에서 다시 시도
        if time.time() < getattr(self, '_cooldown_until', 0.0):
            return

        if self.is_processing:
            return
        self.is_processing = True

        try:
            self.get_logger().info("🎤 주변 소리를 듣는 중...")
            recording = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate, channels=1, dtype='float32'
            )
            sd.wait()

            rms = float(np.sqrt(np.mean(recording ** 2)))
            if rms <= self.threshold:
                self.get_logger().info(f"💤 조용함 (볼륨: {rms:.5f})")
                time.sleep(0.2)
                return

            if not self.client:
                self.get_logger().error("❌ Gemini client가 없습니다 (API KEY 확인).")
                time.sleep(1.0)
                return

            self.get_logger().info(f"🔊 소리 감지! (볼륨: {rms:.5f}) - STT 분석 시작")
            text = self._stt_with_gemini(recording)
            if text:
                raw_text = text.strip()
                self.get_logger().info(f"📝 인식 결과: '{raw_text}'")
                self.pub.publish(String(data=raw_text))

                # cooldown 설정(블로킹 sleep 대신 timestamp로 처리)
                self._cooldown_until = time.time() + float(self.cooldown_sec)
                self.get_logger().info(f"⏳ cooldown {self.cooldown_sec:.1f}s (다음 명령 인식 대기)")

        except Exception as e:
            self.get_logger().error(f"❌ Ear 오류: {e}")
            time.sleep(2.0)
        finally:
            self.is_processing = False

    def _stt_with_gemini(self, recording: np.ndarray):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                wav.write(tmp_path, self.samplerate, recording)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction="당신은 로봇의 귀입니다. 오디오를 듣고 한국어 텍스트로만 정확히 변환하세요. 부가 설명은 절대 하지 마세요.",
                    temperature=0.0,
                ),
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")]
            )
            return getattr(response, "text", None)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


def main(args=None):
    rclpy.init(args=args)
    node = EarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 Ear 종료")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()