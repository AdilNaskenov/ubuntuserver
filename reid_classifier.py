"""
Обёртка над обученным классификатором особей коров (изначально .pth,
здесь используется уже сконвертированная ONNX-версия — см.
export_reid_to_onnx.py). Предсказывает, какая именно корова на кропе,
ДО того как кроп уйдёт на ручную разметку в Re-ID группу — чтобы
предложить готовую подсказку вместо того, чтобы человек листал весь список.

ВАЖНО: обучена только на классе "cow" (SOURCE_DIR = reid_dataset/cow в
training-скрипте) — для "horse" подсказок не будет, пока не появится
отдельная модель для лошадей.
"""
import json
import logging
import cv2
import numpy as np
import onnxruntime as ort

import config

logger = logging.getLogger(__name__)


class CowReIDClassifier:
    def __init__(self, model_path=config.COW_REID_MODEL_PATH,
                 classes_path=config.COW_REID_CLASSES_PATH):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        with open(classes_path, "r", encoding="utf-8") as f:
            # {"0": "1_Буренка", "1": "2_Ромашка", ...} — ключ = индекс класса,
            # значение = имя папки в reid_dataset/cow (формат "{id}_{имя}",
            # см. save_crop_to_reid в animals_store.py). Имя папки может быть
            # "очищено" от спецсимволов при сохранении, поэтому для отображения
            # лучше брать каноничное имя из animals_db по animal_id, а не
            # парсить эту строку напрямую (см. predict()).
            self.idx_to_folder = json.load(f)

        # Точно та же нормализация, что и в training-скрипте
        # (transforms.Normalize с ImageNet-статистикой).
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr):
        img = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        # torchvision.datasets.ImageFolder читает через PIL — то есть в RGB,
        # а не BGR (как отдаёт OpenCV) — обязательно конвертируем, иначе
        # каналы будут перепутаны и модель будет давать случайные предсказания.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0).astype(np.float32)

    def predict(self, crop_bgr):
        """
        Возвращает (animal_id, folder_name, confidence) для лучшего предсказания.
        animal_id — строка вида "1", "2" и т.д. (совпадает с ключами в animals_db["cow"]).
        confidence — от 0 до 1.
        """
        blob = self._preprocess(crop_bgr)
        logits = self.session.run(None, {self.input_name: blob})[0][0]

        # Softmax вручную — ONNX-модель отдаёт сырые логиты, не вероятности.
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        folder_name = self.idx_to_folder.get(str(best_idx), "")

        # Папка называется "{animal_id}_{имя}" — нам нужен именно animal_id,
        # чтобы найти каноничное имя в animals_db (folder_name мог быть
        # "очищен" от спецсимволов при сохранении и не совпадать 1-в-1).
        animal_id = folder_name.split("_", 1)[0] if folder_name else None

        return animal_id, folder_name, confidence


# Пытаемся загрузить классификатор, но НЕ падаем, если файлов ещё нет —
# например, модель для лошадей ещё не обучена, или ONNX ещё не сконвертирован.
# В этом случае подсказки просто не будет, кроп уйдёт на обычную ручную разметку.
try:
    cow_reid_classifier = CowReIDClassifier()
    logger.info("✅ Re-ID классификатор коров загружен.")
except Exception as e:
    cow_reid_classifier = None
    logger.warning(f"⚠️ Re-ID классификатор коров не загружен ({e}) — "
                    f"подсказок по кропам не будет, только ручная разметка.")
