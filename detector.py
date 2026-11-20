import cv2
import numpy as np
from ultralytics import YOLO

class RoadDamageDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.33, imgsz=800):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        # CLAHE filter matching the exact training preprocessing pipeline
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.class_names = {
            0: "D00 - Longitudinal Crack",
            1: "D10 - Transverse Crack",
            2: "D20 - Alligator Crack",
            3: "D40 - Pothole"
        }

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def predict_frame(self, frame: np.ndarray):
        processed_frame = self.preprocess(frame)
        results = self.model.predict(
            source=processed_frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": cls_id,
                "label": self.class_names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        annotated_frame = results.plot()
        return annotated_frame, detections

if __name__ == "__main__":
    # Quick usage example for Backend integration
    detector = RoadDamageDetector(model_path="best.pt", conf_threshold=0.33, imgsz=800)
    print("RoadDamageDetector initialized successfully!")
