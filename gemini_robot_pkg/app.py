import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, threading, json, base64, os, time, re
from flask import Flask, render_template, Response, request, jsonify
from ament_index_python.packages import get_package_share_directory
import google.generativeai as genai

# --- [경로 및 Flask 설정] ---
try:
    share_dir = get_package_share_directory('gemini_robot_pkg')
    template_dir = os.path.join(share_dir, 'templates')
except:
    template_dir = "templates"

app = Flask(__name__, template_folder=template_dir)
bridge = CvBridge()
latest_frame = None
latest_sequence = [] 
frame_lock = threading.Lock()

# --- [Gemini AI 설정] ---
# API 키 설정
genai.configure(api_key="AIzaSyC0C6PjMpVlROy6Z9BDdWpiwbWcP_jwhBs")

# [수정] 모델 이름을 가장 표준적인 'gemini-1.5-flash'로 설정합니다.
# 404 에러가 지속될 경우 터미널에서 'pip install -U google-generativeai'를 꼭 실행해 주세요.
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', 
    generation_config={"response_mime_type": "application/json"}
)

# --- [ROS 2 구독 노드] ---
class WebSubscriber(Node):
    def __init__(self):
        super().__init__('web_subscriber')
        # 영상 구독 (eye_ui.py로부터 수신)
        self.create_subscription(Image, '/eye/processed_frame', self.img_cb, 10)
        # 작업 순서 구독 (brain_node.py로부터 수신)
        self.create_subscription(String, '/brain/normalized_coords', self.brain_cb, 10)
        
    def img_cb(self, data):
        global latest_frame
        with frame_lock:
            latest_frame = bridge.imgmsg_to_cv2(data, 'bgr8')

    def brain_cb(self, msg):
        global latest_sequence
        try:
            latest_sequence = json.loads(msg.data)
            self.get_logger().info(f"📥 Brain으로부터 {len(latest_sequence)}개의 작업 단계 수신 완료")
        except Exception as e:
            self.get_logger().error(f"Brain 데이터 파싱 에러: {e}")

# --- [Flask 라우트 설정] ---

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    # JPEG 품질 조절로 전송 속도 최적화
                    _, buf = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.04) # 약 25 FPS 유지
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sequence')
def get_sequence():
    global latest_sequence
    return jsonify(latest_sequence)

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # 클라이언트로부터 이미지 데이터 수신
        img_b64 = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        
        prompt = """너는 전기 회로 조립 튜터야. 이미지는 실제 장치 위에 사용자가 가상으로 배선을 그린 사진이야. 
                  [분석 요청]:
                  1. 사용자가 그린 선이 회로도의 논리(예: Relay 4번 -> Timer 6번)와 일치하는지 판별해줘.
                  2. 단자 라벨(예: relay4 3, timer7 8)을 참고하여 정확한 위치인지 확인해줘.
                  [응답 형식]:
                  반드시 아래의 JSON 형식으로만 답변해줘.
                  {"result": "OK" 또는 "NG", "reason": "한글 설명"}"""
        
        # AI 분석 요청
        response = model.generate_content([
            prompt, 
            {'mime_type': 'image/jpeg', 'data': img_bytes}
        ])
        
        # [수정] 응답 텍스트에서 혹시 모를 마크다운 코드 블록(```json) 제거 후 파싱
        clean_text = re.sub(r'```json|```', '', response.text).strip()
        return jsonify(json.loads(clean_text))
        
    except Exception as e:
        # 에러 발생 시 상세 로그 출력
        print(f"Verify Error: {str(e)}")
        return jsonify({"result": "NG", "reason": f"AI 분석 중 오류 발생: {str(e)}"})

# --- [메인 실행부] ---
def main():
    # ROS 2 노드 스레드 실행
    ros_thread = threading.Thread(
        target=lambda: (rclpy.init(), rclpy.spin(WebSubscriber()), rclpy.shutdown()), 
        daemon=True
    )
    ros_thread.start()
    
    print("🚀 Flask 서버 가동 중: http://localhost:5000")
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

if __name__ == '__main__':
    main()
