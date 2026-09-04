"""
Точка входа. Собирает Dispatcher из роутеров, запускает основной цикл
обработки видео и поллинг Telegram-бота.
"""
import logger_setup  # noqa: F401 — вызывает logging.basicConfig() до первого getLogger()

import asyncio
import logging
import time
import cv2
from aiogram import Dispatcher
from aiogram.types import BufferedInputFile

import config
import camera
from bot_instance import bot
from motion import motion_detector
from processing import process_frame
from reid_router import send_crop_for_labeling
import commands_router
import reid_router

logger = logging.getLogger(__name__)

# ВАЖНО: порядок подключения роутеров важен! commands_router содержит
# конкретные команды (/photo, /testurl, /start), а reid_router — в том
# числе хендлер-ловушку без фильтров (handle_name_reply), которая должна
# быть проверена ПОСЛЕДНЕЙ, иначе она перехватит на себя вообще любой текст,
# включая команды.
dp = Dispatcher()
dp.include_router(commands_router.router)
dp.include_router(reid_router.router)


async def video_processing_loop():
    logger.info("🎥 Подключение к RTSP...")
    stream = camera.RTSPStreamReader(config.STREAM_URL)
    camera.set_current_stream_reader(stream)  # чтобы /photo мог взять кадр из этого же потока

    last_crop_time = 0
    last_raw_time = 0
    last_alarm_time = 0
    last_forced_check_time = 0
    last_inference_time = 0

    skipped_frames_count = 0

    while True:
        try:
            ret, frame = stream.read()
            if not ret or frame is None:
                await asyncio.sleep(0.05)
                continue

            current_time = time.time()

            # Дешёвая проверка движения — всегда на низком разрешении (ch0),
            # ей высокое разрешение не нужно, она и так уменьшает кадр до 320x240.
            has_motion = await asyncio.to_thread(motion_detector.detect_motion, frame)
            force_check = (current_time - last_forced_check_time) >= config.FORCE_CHECK_INTERVAL
            should_trigger = has_motion or force_check

            if not should_trigger:
                skipped_frames_count += 1
                await asyncio.sleep(0.1)
                continue

            # Триггер есть, но проверяем отдельный кулдаун именно на инференс —
            # чтобы не бомбить stream1 множеством подключений при затяжном движении.
            if current_time - last_inference_time < config.INFERENCE_COOLDOWN:
                await asyncio.sleep(0.1)
                continue

            last_inference_time = current_time
            if force_check:
                last_forced_check_time = current_time

            if skipped_frames_count > 0:
                logger.info(f"⏭️  Пропущено {skipped_frames_count} кадров без движения "
                            f"(YOLO не запускался)")
                skipped_frames_count = 0

            # Тянем свежий кадр с высокого разрешения (stream1) ПРЯМО СЕЙЧАС
            # и детектируем именно на нём — а не на низком ch0 с последующим
            # апскейлом. Модель обучена на 1280p, а 2560x1440 → 1280 (downscale)
            # сохраняет реальные детали, в отличие от апскейла 640 → 1280,
            # который их просто не может дорисовать.
            detection_frame = await asyncio.to_thread(camera.grab_high_res_frame)

            if detection_frame is None:
                logger.warning("⚠️ stream1 не ответил — детектирую на низком "
                                "разрешении (ch0) как fallback.")
                detection_frame = frame

            annotated_frame, detected_targets, last_crop_time, last_raw_time, crops_to_label = \
                await asyncio.to_thread(
                    process_frame, detection_frame, last_crop_time, last_raw_time,
                    config.CROP_COOLDOWN, config.RAW_COOLDOWN
                )

            for crop_path, class_name, full_frame_path in crops_to_label:
                asyncio.create_task(send_crop_for_labeling(crop_path, class_name, full_frame_path))

            current_time = time.time()

            if detected_targets and (current_time - last_alarm_time >= config.ALARM_COOLDOWN):
                last_alarm_time = current_time
                unique_targets = list(set(detected_targets))
                logger.info(f"🚨 Найдено: {unique_targets}. Отправка в ТГ...")

                _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                input_file = BufferedInputFile(buffer.tobytes(), filename="alarm.jpg")

                asyncio.create_task(
                    bot.send_photo(
                        chat_id=config.MY_CHAT_ID,
                        photo=input_file,
                        caption=f"⚠️ **Обнаружен объект!**\nТип: `{', '.join(unique_targets)}`",
                        parse_mode="Markdown"
                    )
                )

            await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Ошибка в цикле: {e}")
            await asyncio.sleep(1)


async def main():
    logger.info("🤖 Запуск скрипта...")
    asyncio.create_task(video_processing_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Остановлено.")
