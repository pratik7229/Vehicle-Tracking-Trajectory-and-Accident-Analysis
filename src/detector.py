from ultralytics import YOLO
import src.config as cf


class YOLODetector:
    def __init__(self):
        self.model = YOLO(cf.MODEL_PATH).to(cf.DEVICE)

    def get_detection(self, frame):
        # results[0].boxes.data -> [x1, y1, x2, y2, confidence, class]
        results = self.model.predict(frame, conf=cf.CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            
            if int(class_id) in cf.VEHICLE_CLASSES:
                # Deep SORT expects [left, top, w, h]
                w = x2 - x1
                h = y2 - y1
                # Format: ([x,y,w,h], confidence, class_id)
                detections.append(([int(x1), int(y1), int(w), int(h)], score, int(class_id)))
        
        return detections


