"""
Обработка одного кадра: детекция через YOLO, отрисовка рамок для алерта,
сохранение кропов/raw-кадров для датасета.
"""
import os
import time
from datetime import datetime
import cv2

import config
from detection import detector


def process_frame(frame, last_crop_time, last_raw_time, crop_cooldown, raw_cooldown):
    """
    ВАЖНО: 'frame' здесь — это уже ТОТ кадр, на котором нужно детектировать
    (см. main.py: обычно это свежий кадр с stream1, полученный по событию
    движения; если stream1 недоступен — fallback на low-res ch0). Никакого
    отдельного пересчёта координат между разными источниками не нужно —
    детекция и сохранение кропов/raw идут на одном и том же кадре.
    """
    annotated_frame = frame.copy()
    boxes, confs, class_ids = detector.detect(frame)

    detected_targets = []
    crops_to_label = []
    current_time = time.time()

    can_save_crops = current_time - last_crop_time >= crop_cooldown
    can_save_raw = current_time - last_raw_time >= raw_cooldown
    crops_saved_this_frame = False

    for box, conf, cid in zip(boxes, confs, class_ids):
        x, y, w, h = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        label_name = config.CLASS_NAMES.get(cid, "Object")
        detected_targets.append(label_name)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label_name} {conf:.2f}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if can_save_crops:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_path = os.path.join(config.DATASET_CROPS_DIR, f"crop_{label_name}_{timestamp}.jpg")
                cv2.imwrite(crop_path, crop)
                crops_saved_this_frame = True

                if label_name in config.REID_CLASSES:
                    full_frame_path = os.path.join(
                        config.PENDING_FULL_FRAMES_DIR, f"frame_{label_name}_{timestamp}.jpg"
                    )
                    cv2.imwrite(full_frame_path, frame)
                    crops_to_label.append((crop_path, label_name, full_frame_path))

    if crops_saved_this_frame:
        last_crop_time = current_time

    if detected_targets and can_save_raw:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(os.path.join(config.DATASET_RAW_DIR, f"raw_{timestamp}.jpg"), frame)
        last_raw_time = current_time

    return annotated_frame, detected_targets, last_crop_time, last_raw_time, crops_to_label
