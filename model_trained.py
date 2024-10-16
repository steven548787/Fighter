import threading
import time
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU，只允許使用 CPU

from inference import get_model
class Instruction:
    def __init__(self):
        self.model = get_model("gensture/5", api_key="drc3S7lkxs5vf4DAWWi0")
        self.cap = cv2.VideoCapture(0)
        self.cls = None
        self.running = True

    def detect_gesture(self):
        while self.running:
            print("偵測中...")
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取影像")
                break
            elif ret:
                results = self.model.infer(image=frame)
                print(results)  # 確認推論結果

            # 檢查推論結果是否有效且包含預測
            if results and isinstance(results, list) and len(results) > 0:
                if hasattr(results[0], 'predictions') and len(results[0].predictions) > 0:
                    # 提取 class_name
                    self.cls = results[0].predictions[0].class_name
                    print(f"偵測到的類別名稱: {self.cls}")
                else:
                    print("沒有偵測到物體")
                    self.cls = None  # 若無預測結果，將類別設為 None
            else:
                print("推論結果無效或為空")
                self.cls = None  # 若推論結果無效，將類別設為 None

                break
            #time.sleep(0.1)  # 避免過多資源消耗

    def start_detection(self):
        threading.Thread(target=self.detect_gesture, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def get_gesture(self):
        return self.cls
