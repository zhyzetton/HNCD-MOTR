from ultralytics import YOLO
import numpy as np

class ProposalGenerator:
    def __init__(self, model_path='yolov8x.pt', device='cuda'):
        print(f"Loading YOLOv8 from {model_path}...")
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame, conf_thres=0.5):
        # 0: person, 2: car, 5: bus, 7: truck
        results = self.model(frame, classes=[0, 2, 5, 7], conf=conf_thres, verbose=False, device=self.device)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = results[0].boxes.conf.cpu().numpy() # (N,)
        classes = results[0].boxes.cls.cpu().numpy() # (N,)
        
        # 你的 HNCD-MOTR 可能需要 (cx, cy, w, h) 格式，这里可能需要转换
        # 这里返回最原始的，在 tracker.py 里转
        return boxes, scores, classes