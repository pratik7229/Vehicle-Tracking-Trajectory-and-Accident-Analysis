import cv2
import numpy as np
from collections import defaultdict, deque

from src.config import *
from src.tracker import ObjectTracker
from src.detector import *


# Store trajectory history per track
track_data = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))


def compute_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def main(video_path, track_type="deep"):
    cap = cv2.VideoCapture(video_path)
    detector = YOLODetector()
    tracker = ObjectTracker(track_type=track_type)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # ---------------- DETECTION ----------------
        detections = detector.get_detection(frame)

        tracks = tracker.update(detections, frame)

        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]

            w = x2 - x1
            h = y2 - y1

            if w < 30 or h < 30:
                continue

            center = compute_center((x1, y1, x2, y2))
            track_data[tid].append(center)

            # velocity
            velocity = (0, 0)
            if len(track_data[tid]) >= 2:
                (x_prev, y_prev) = track_data[tid][-2]
                velocity = (center[0] - x_prev, center[1] - y_prev)

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # trajectory
            for i in range(1, len(track_data[tid])):
                pt1 = (int(track_data[tid][i - 1][0]), int(track_data[tid][i - 1][1]))
                pt2 = (int(track_data[tid][i][0]), int(track_data[tid][i][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        # ---------------- DISPLAY ----------------
        cv2.imshow("Multi Object Tracker (Fixed)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path1 = "/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/videos/_-XReeuhpiE_00.mp4"
    path2 = "/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/videos/_5e--tkpUck_00.mp4"
    path3 = "/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/videos/_9tp71S2Gwg_00.mp4"
    path4 = "/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/videos/-AztVDZ6cEE_00.mp4"
    path5 = "/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/videos/0ThWw_efieo_01.mp4"
    
    #main("/Users/pratik/Documents/Projects/kaggleCompetation_data/Accident@CVPR/sim_dataset/videos/head-on/Town03_head-on_clear_01.mp4",  track_type="byte")
    #main(path)
    main(path3, track_type="byte")