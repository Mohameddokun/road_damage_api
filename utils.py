import cv2

CLASS_COLORS = {
    "D00": (255, 191, 0),   # Longitudinal Crack - Cyan/Blue
    "D10": (0, 215, 255),   # Transverse Crack   - Yellow
    "D20": (0, 140, 255),   # Alligator Crack    - Orange
    "D40": (0, 0, 255),     # Pothole            - Red
}

SEVERITY_WEIGHTS = {
    "D00": 1,
    "D10": 1,
    "D20": 3,
    "D40": 4,
}


def assess_road_condition(counts: dict[str, int]) -> str:
    score = sum(counts.get(cls, 0) * weight for cls, weight in SEVERITY_WEIGHTS.items())
    if score <= 2:
        return "Good"
    if score <= 8:
        return "Fair"
    return "Dangerous"


def draw_annotations(frame, detections: list[dict], fps: float, condition: str):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls = det["class_name"]
        conf = det["conf"]
        tid = det.get("track_id")

        color = CLASS_COLORS.get(cls, (0, 255, 0))
        label = f"#{tid} {cls} {conf:.2f}" if tid is not None else f"{cls} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

    # Top-left HUD overlay
    cond_color = {"Good": (0, 200, 0), "Fair": (0, 165, 255), "Dangerous": (0, 0, 255)}[condition]
    cv2.putText(annotated, f"FPS: {fps:.1f} | Road: {condition}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, cond_color, 2)
    return annotated