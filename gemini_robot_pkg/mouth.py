import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import time
from gtts import gTTS
import pygame
import tempfile
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

class MouthNode(Node):
    def __init__(self):
        super().__init__('mouth_node')
        
        # 1. 오디오 믹서 초기화 (기존 TTS 클래스 로직)
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
        # 2. 통신 설정: Brain으로부터 말할 내용을 전달받음
        self.subscription = self.create_subscription(
            String,
            '/mouth/speech_text',
            self.speech_callback,
            10
        )
        
        self.get_logger().info("👄 입(Mouth) 노드가 준비되었습니다. 명령을 기다립니다.")

    def speech_callback(self, msg):
        """Brain 노드로부터 받은 텍스트를 음성으로 출력"""
        text = msg.data
        self.get_logger().info(f"음성 출력 시도: {text}")
        self.text2speech(text)

    def text2speech(self, text, lang='ko'):
        """기존에 작성하신 TTS 변환 및 재생 로직"""
        if not text.strip():
            return
            
        temp_filename = None
        try:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                temp_filename = temp_mp3.name
                tts.save(temp_filename)

            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            
            pygame.mixer.music.unload()
            
        except Exception as e:
            self.get_logger().error(f"TTS 재생 중 오류 발생: {e}")
        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except PermissionError:
                    pass

def main(args=None):
    rclpy.init(args=args)
    node = MouthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
