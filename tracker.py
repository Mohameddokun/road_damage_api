import numpy as np

DAMAGE_CLASSES = ("D00", "D10", "D20", "D40")


def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


class DamageTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 12, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits  # Require N frames before counting to reduce false positives
        self.next_id = 1
        self.tracks = {}  # track_id -> {bbox, class_name, lost, hits}
        self.counted_ids = {cls: set() for cls in DAMAGE_CLASSES}

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: [{"bbox": [x1, y1, x2, y2], "class_name": "D40", "conf": 0.65}, ...]
        Returns detections enriched with a persistent `track_id`.
        """
        matched_tracks = set()
        matched_dets = set()
        tracked_outputs = []

        # 1. Build candidate (iou, track_id, det_idx) pairs for same-class matches
        pairs = []
        for tid, track in self.tracks.items():
            for idx, det in enumerate(detections):
                if det["class_name"] == track["class_name"]:
                    iou = compute_iou(track["bbox"], det["bbox"])
                    if iou >= self.iou_threshold:
                        pairs.append((iou, tid, idx))

        # 2. Greedy match by highest IoU
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, tid, idx in pairs:
            if tid in matched_tracks or idx in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(idx)

            det = detections[idx]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["lost"] = 0
            self.tracks[tid]["hits"] += 1

            if self.tracks[tid]["hits"] >= self.min_hits and det["class_name"] in self.counted_ids:
                self.counted_ids[det["class_name"]].add(tid)

            tracked_outputs.append({**det, "track_id": tid})

        # 3. Register new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "lost": 0,
                    "hits": 1,
                }
                if self.min_hits <= 1 and det["class_name"] in self.counted_ids:
                    self.counted_ids[det["class_name"]].add(tid)
                tracked_outputs.append({**det, "track_id": tid})

        # 4. Age out disappeared tracks
        expired = []
        for tid in self.tracks:
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        return tracked_outputs

    def get_counts(self) -> dict[str, int]:
        return {cls: len(ids) for cls, ids in self.counted_ids.items()}