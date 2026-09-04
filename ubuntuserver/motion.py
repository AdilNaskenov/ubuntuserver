"""
Дешёвая проверка "изменилось ли что-то в кадре" — чтобы не гонять тяжёлый
YOLO на каждый кадр, а только когда реально есть движение.
"""
import cv2
import numpy as np

import config


class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect_motion(self, frame):
        small = cv2.resize(frame, config.MOTION_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, config.MOTION_BLUR_KSIZE, 0)
        # Часы камеры в углу меняются каждую секунду — исключаем эту зону,
        # иначе тикающий таймштамп сам по себе считался бы "движением".
        gray[0:25, 0:140] = 0

        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        diff = cv2.absdiff(self.prev_gray, gray)
        changed_pixels = np.count_nonzero(diff > config.MOTION_DIFF_THRESHOLD)
        total_pixels = diff.size
        changed_fraction = changed_pixels / total_pixels

        self.prev_gray = gray

        return changed_fraction >= config.MOTION_MIN_CHANGED_FRACTION


# Единственный экземпляр на весь проект — хранит состояние (prev_gray)
# между вызовами, поэтому важно, чтобы все части кода использовали именно
# этот общий объект, а не создавали свой.
motion_detector = MotionDetector()
