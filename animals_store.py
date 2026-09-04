"""
Реестр известных особей (animals.json) и файловые операции для Re-ID:
перенос кропов в reid_dataset/, в misclassified/, удаление неиспользованных
полных кадров. Плюс общее состояние ожидающих разметки кропов, используемое
и в основном цикле, и в хендлерах Telegram.
"""
import asyncio
import json
import os

import config


def load_animals():
    if os.path.exists(config.ANIMALS_DB_PATH):
        with open(config.ANIMALS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_animals(data):
    with open(config.ANIMALS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- Общее состояние на весь проект ---
# animals_db:       {"cow": {"1": "Бурёнка", "2": "Ромашка"}, "horse": {...}}
# pending_crops:    message_id (в REID-группе) -> (class_name, crop_path, full_frame_path)
# waiting_for_name: chat_id -> (class_name, message_id) — ждём ответ с кличкой новой особи
animals_db = load_animals()
pending_crops = {}
waiting_for_name = {}


async def save_crop_to_reid(class_name, animal_id, animal_name, crop_path):
    """Переносит кроп в reid_dataset/<класс>/<id_кличка>/, готовое для обучения Re-ID."""
    safe_name = "".join(c for c in animal_name if c.isalnum() or c in ("_", "-")) or "unnamed"
    dest_dir = os.path.join(config.REID_DATASET_DIR, class_name, f"{animal_id}_{safe_name}")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(crop_path))
    await asyncio.to_thread(os.rename, crop_path, dest_path)


async def discard_pending_full_frame(full_frame_path):
    """Кроп разметился нормально (особь выбрана) — полный кадр для переразметки
    больше не нужен, удаляем, чтобы не копить лишнее на диске."""
    try:
        await asyncio.to_thread(os.remove, full_frame_path)
    except FileNotFoundError:
        pass


async def save_crop_as_misclassified(predicted_class, crop_path, full_frame_path):
    """YOLO ошиблась с классом — переносим кроп И полный кадр в отдельные папки
    на переразметку, а не в reid_dataset, чтобы не засорять Re-ID датасет
    неверными примерами. Возвращает путь к полному кадру в его новом месте
    (для пересылки пользователю)."""
    crop_dest_dir = os.path.join(config.MISCLASSIFIED_DIR, f"predicted_{predicted_class}")
    os.makedirs(crop_dest_dir, exist_ok=True)
    crop_dest_path = os.path.join(crop_dest_dir, os.path.basename(crop_path))
    await asyncio.to_thread(os.rename, crop_path, crop_dest_path)

    frame_dest_dir = os.path.join(config.MISCLASSIFIED_FULL_FRAMES_DIR, f"predicted_{predicted_class}")
    os.makedirs(frame_dest_dir, exist_ok=True)
    frame_dest_path = os.path.join(frame_dest_dir, os.path.basename(full_frame_path))
    await asyncio.to_thread(os.rename, full_frame_path, frame_dest_path)

    return frame_dest_path
