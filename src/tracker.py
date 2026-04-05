import numpy as np
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
import src.config as cf


class ObjectTracker:
    def __init__(self, track_type="deep"):
        self.track_type = track_type

        if track_type == "deep":
            self.tracker = DeepSort(
                max_age=cf.MAX_AGE,
                n_init=cf.N_INT,
                max_cosine_distance=cf.MAX_COSINE_DISTANCE,
                embedder=None,  # important
            )

        elif track_type == "byte":
            self.tracker = sv.ByteTrack()

        else:
            raise ValueError("track_type must be 'deep' or 'byte'")

    def update(self, detections, frame):
        if self.track_type == "deep":
            return self._update_deepsort(detections, frame)
        else:
            return self._update_bytetrack(detections, frame)

    # ---------------- DeepSORT ----------------
    def _update_deepsort(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                continue

            x1, y1, x2, y2 = map(int, ltrb)

            results.append({
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2]
            })

        return results

    # ---------------- ByteTrack ----------------
    def _update_bytetrack(self, detections, frame):
        if len(detections) == 0:
            return []

        xyxy = []
        confidences = []
        class_ids = []

        for det in detections:
            (x, y, w, h), score, cls = det

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            if w < 30 or h < 30:
                continue

            xyxy.append([x1, y1, x2, y2])
            confidences.append(score)
            class_ids.append(cls)

        if len(xyxy) == 0:
            return []

        detections_sv = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        tracks = self.tracker.update_with_detections(detections_sv)

        results = []
        for i in range(len(tracks)):
            tid = tracks.tracker_id[i]
            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, tracks.xyxy[i])

            results.append({
                "track_id": int(tid),
                "bbox": [x1, y1, x2, y2]
            })

        return results