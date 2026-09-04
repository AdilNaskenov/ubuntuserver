"""
Команды для ручного взаимодействия с ботом: /photo (текущий кадр),
/testurl (проверка конкретного RTSP-пути), /start.

ВАЖНО: этот роутер должен подключаться в main.py ПЕРЕД reid_router —
там есть хендлер-ловушка без фильтров (handle_name_reply), которая обязана
идти последней, иначе перехватит команды на себя.
"""
import asyncio
import cv2
from aiogram import Router, types
from aiogram.filters import CommandStart, Command
from aiogram.types import BufferedInputFile
from aiogram.utils.keyboard import ReplyKeyboardBuilder

import config
import camera

router = Router()


def build_main_keyboard():
    """Обычная (не inline) клавиатура с кнопкой быстрого получения фото —
    видна всегда внизу чата, не привязана к конкретному сообщению."""
    builder = ReplyKeyboardBuilder()
    builder.button(text="📷 Текущее фото")
    return builder.as_markup(resize_keyboard=True)


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "🤖 Бот запущен на Ubuntu Server (Pentium E5700)!\n\n"
        "Команды:\n"
        "/photo — текущий кадр с камеры\n"
        "/testurl <имя> — проверить конкретный RTSP-путь (ch0, ch1, stream1)",
        reply_markup=build_main_keyboard()
    )


@router.message(Command("photo"))
async def cmd_photo(message: types.Message):
    """
    Присылает самый свежий кадр с ОСНОВНОГО потока (того же, что и для
    YOLO/фильтра движения) — без нового подключения к камере, берёт кадр
    из уже открытого RTSPStreamReader.
    """
    stream_reader = camera.get_current_stream_reader()

    if stream_reader is None:
        await message.reply("⚠️ Поток с камеры ещё не запущен, попробуй чуть позже.")
        return

    ret, frame = stream_reader.read()
    if not ret or frame is None:
        await message.reply("⚠️ Не удалось получить кадр с камеры прямо сейчас.")
        return

    h, w = frame.shape[:2]
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    input_file = BufferedInputFile(buffer.tobytes(), filename="photo.jpg")
    await message.reply_photo(photo=input_file, caption=f"📷 Текущий кадр ({w}x{h})")


@router.message(lambda m: m.text == "📷 Текущее фото")
async def handle_photo_button(message: types.Message):
    # Кнопка — просто вызывает ту же логику, что и команда /photo
    await cmd_photo(message)


@router.message(Command("testurl"))
async def cmd_testurl(message: types.Message):
    """
    Подключается на короткое время к одному из кандидатов CANDIDATE_STREAMS,
    берёт один кадр и присылает в чат — чтобы сравнить разрешение/качество
    разных RTSP-путей камеры прямо в Telegram, без VLC.

    Использование: /testurl ch1  (или /testurl без аргумента — покажет список)
    """
    args = message.text.split(maxsplit=1)

    if len(args) < 2:
        names = ", ".join(config.CANDIDATE_STREAMS.keys())
        await message.reply(f"Укажи имя потока для проверки. Доступные: {names}\n"
                             f"Пример: /testurl ch1")
        return

    name = args[1].strip()
    url = config.CANDIDATE_STREAMS.get(name)
    if url is None:
        names = ", ".join(config.CANDIDATE_STREAMS.keys())
        await message.reply(f"Не знаю такой поток: '{name}'. Доступные: {names}")
        return

    await message.reply(f"⏳ Подключаюсь к {name} ({url})...")

    frame = await asyncio.to_thread(camera.grab_single_frame, url)

    if frame is None:
        await message.reply(f"❌ Не удалось получить кадр с '{name}' за отведённое время. "
                             f"Либо неверный путь, либо камера не отвечает по нему.")
        return

    h, w = frame.shape[:2]
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    input_file = BufferedInputFile(buffer.tobytes(), filename=f"{name}.jpg")
    await message.reply_photo(photo=input_file, caption=f"📷 {name}: {w}x{h}\n{url}")
