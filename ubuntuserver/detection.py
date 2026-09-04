"""
Движок инференса YOLOv8 через ONNX Runtime — детекция cow/horse/person.
"""
import cv2
import numpy as np
import onnxruntime as ort

import config


class YOLOv8ONNX:
    def __init__(self, model_path, conf_thresh=0.60, iou_thresh=0.45):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.img_size = (1280, 1280)

        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name

        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh

    def preprocess(self, img):
        """
        Letterbox-препроцессинг — ТАК ЖЕ, как ultralytics обрабатывает кадры
        на тренировке: сохраняем пропорции кадра, добиваем до квадрата серыми
        полосами (114,114,114 — стандартный паддинг-цвет ultralytics), а НЕ
        тупо растягиваем прямоугольник в квадрат через cv2.resize напрямую.

        Без этого модель на инференсе видела бы искажённые (сплющенные или
        растянутые) пропорции объектов — то, чего никогда не было в
        тренировочных данных, где letterbox использовался по умолчанию.
        """
        h0, w0 = img.shape[:2]
        target_w, target_h = self.img_size

        # Один и тот же коэффициент для обеих осей — так пропорции не искажаются.
        scale = min(target_w / w0, target_h / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Паддинг делим поровну на обе стороны, чтобы объект оказался по
        # центру квадрата — так же, как это делает ultralytics letterbox.
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # Эти три числа понадобятся в detect(), чтобы правильно пересчитать
        # координаты боксов обратно в систему координат ОРИГИНАЛЬНОГО кадра
        # (снять паддинг и обратный масштаб).
        letterbox_info = {"scale": scale, "pad_left": left, "pad_top": top}
        return blob, letterbox_info

    def detect(self, img):
        blob, letterbox_info = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]

        predictions = np.squeeze(outputs).T

        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]

        if len(predictions) == 0:
            return [], [], []

        raw_class_ids = np.argmax(predictions[:, 4:], axis=1)

        scale = letterbox_info["scale"]
        pad_left = letterbox_info["pad_left"]
        pad_top = letterbox_info["pad_top"]

        boxes, confidences, class_ids = [], [], []
        for pred, score, cid in zip(predictions, scores, raw_class_ids):
            if cid in config.TARGET_CLASSES:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

                # Снимаем паддинг и обратный масштаб — координаты сети были
                # в системе letterbox-квадрата (1280x1280), а не оригинала.
                cx = (cx - pad_left) / scale
                cy = (cy - pad_top) / scale
                w = w / scale
                h = h / scale

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                width = int(w)
                height = int(h)

                boxes.append([x1, y1, width, height])
                confidences.append(float(score))
                class_ids.append(int(cid))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        final_boxes, final_confs, final_classes = [], [], []
        if len(indices) > 0:
            for idx in indices.flatten():
                final_boxes.append(boxes[idx])
                final_confs.append(confidences[idx])
                final_classes.append(class_ids[idx])

        return final_boxes, final_confs, final_classes


# Единственный экземпляр детектора на весь проект — грузится один раз при
# импорте модуля, дальше просто используется (в т.ч. в других модулях
# через `from detection import detector`).
detector = YOLOv8ONNX(config.MODEL_PATH)
