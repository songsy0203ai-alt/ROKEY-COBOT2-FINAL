"""
- [코드 기능]: 카메라 영상의 2D 픽셀 좌표와 로봇 작업 평면의 mm 좌표 사이의 
              호모그래피 행렬을 생성하여 저장함.
- [입력(Input)]: 1. 영상 내 4개 지점 마우스 클릭
                2. 터미널 입력 (실제 로봇 X, Y mm 좌표)
- [출력(Output)]: matrix.json (저장 위치: data/calibration/)
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
# nerve.py에 정의된 유틸리티 클래스를 임포트합니다.
from gemini_robot_pkg.nerve import GeminiNerve

class CalibNode(Node):
    def __init__(self):
        super().__init__('calib_node')
        
        self.calib_data_path = os.path.expanduser(
            '~/cobot_ws/src/cobot2_ws/gemini_robot_pkg/data/calibration/matrix.json'
        )
        
        # 파일 저장 로직을 nerve_util에 위임
        self.nerve_util = GeminiNerve(self.calib_data_path)
        
        self.img_points = []
        self.robot_points = []
        
        print("\n--- Calibration 시작 ---")
        print("1. 카메라 창에서 대응하는 4개의 점을 순서대로 클릭하세요.")
        print("2. 클릭 시 터미널에 해당 지점의 로봇 좌표(mm)를 입력하세요.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\n[점 {len(self.img_points) + 1}] 클릭 픽셀 좌표: u={x}, v={y}")
            
            try:
                rx = float(input("로봇 X 좌표 (mm): "))
                ry = float(input("로봇 Y 좌표 (mm): "))
                
                self.img_points.append([x, y])
                self.robot_points.append([rx, ry])
                
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(param, f"P{len(self.img_points)}", (x+10, y+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Calibration Window", param)
                
            except ValueError:
                print("숫자만 입력 가능합니다.")

    def run(self):
        cap = cv2.VideoCapture(6) # 카메라 인덱스 확인 필요
        
        if not cap.isOpened():
            print("에러: 카메라를 열 수 없습니다.")
            return

        cv2.namedWindow("Calibration Window")
        
        while len(self.img_points) < 4:
            ret, frame = cap.read()
            if not ret: break
            
            cv2.imshow("Calibration Window", frame)
            cv2.setMouseCallback("Calibration Window", self.mouse_callback, param=frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC 키
                break

        if len(self.img_points) == 4:
            src_pts = np.array(self.img_points, dtype=np.float32)
            dst_pts = np.array(self.robot_points, dtype=np.float32)
            
            # 호모그래피 행렬 계산
            h_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            
            # 저장 실행
            self.nerve_util.save_calibration(h_matrix)
            print("\n--- 캘리브레이션 완료 및 저장 성공! ---")
            print(f"Matrix:\n{h_matrix}")
        
        cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CalibNode()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()