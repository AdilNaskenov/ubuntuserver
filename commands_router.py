"""
Команды для ручного взаимодействия с ботом: /photo (текущий кадр),
/video (5-сек видео), /relay (переключение Sonoff), /status (нагрузка сервера),
/testurl (проверка конкретного RTSP-пути), /start.

ВАЖНО: этот роутер должен подключаться в main.py ПЕРЕД reid_router —
там есть хендлер-ловушка без фильтров (handle_name_reply), которая обязана
идти последней, иначе перехватит команды на себя.
"""
import asyncio
import os
import tempfile
import cv2
import aiohttp
import psutil
from aiogram import Router, types
from aiogram.filters import CommandStart, Command
from aiogram.types import BufferedInputFile, FSInputFile
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from ewelink_handler import SonoffRelay
import config
import camera

router = Router()

# Настройки для Sonoff (LAN режим или Tasmota)
SONOFF_IP = getattr(config, "SONOFF_IP", "192.168.1.150")


def build_main_keyboard():
    """Главная клавиатура управления камерой, реле и сервером."""
    builder = ReplyKeyboardBuilder()
    builder.button(text="📷 Текущее фото")
    builder.button(text="🎥 Видео 5 сек")
    builder.button(text="⚡ Свет/Реле")
    builder.button(text="📊 Статус сервера")
    builder.adjust(2, 2)  # Раскладка 2х2
    return builder.as_markup(resize_keyboard=True)


def encode_jpeg(frame):
    """Вынос тяжелого JPEG-кодирования из главного asyncio-потока."""
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "🤖 Бот управления фермой и камерой запущен!\n\n"
        "Быстрые команды:\n"
        "/photo — фото с камеры\n"
        "/video — записать 5 секунд видео\n"
        "/relay — переключить реле Sonoff\n"
        "/status — нагрузка процессора и памяти\n"
        "/testurl <имя> — проверить RTSP-поток (ch0, ch1)",
        reply_markup=build_main_keyboard()
    )


@router.message(Command("photo"))
@router.message(lambda m: m.text == "📷 Текущее фото")
async def cmd_photo(message: types.Message):
    stream_reader = camera.get_current_stream_reader()

    if stream_reader is None:
        await message.reply("⚠️ Поток с камеры ещё не запущен, попробуй чуть позже.")
        return

    ret, frame = stream_reader.read()
    if not ret or frame is None:
        await message.reply("⚠️ Не удалось получить кадр с камеры прямо сейчас.")
        return

    h, w = frame.shape[:2]
    # Выполняем сжатие изображения асинхронно
    jpeg_bytes = await asyncio.to_thread(encode_jpeg, frame)
    input_file = BufferedInputFile(jpeg_bytes, filename="photo.jpg")
    await message.reply_photo(photo=input_file, caption=f"📷 Текущий кадр ({w}x{h})")


@router.message(Command("video"))
@router.message(lambda m: m.text == "🎥 Видео 5 сек")
async def cmd_video(message: types.Message):
    """Записывает короткий видеофрагмент (5 секунд) с камеры и отправляет в чат."""
    stream_reader = camera.get_current_stream_reader()
    if stream_reader is None:
        await message.reply("⚠️ Поток с камеры не активен.")
        return

    msg = await message.reply("🎥 Записываю 5 секунд видео...")
    
    # Временный файл под mp4
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "stream_clip.mp4")

    def record_clip():
        ret, sample_frame = stream_reader.read()
        if not ret or sample_frame is None:
            return False
        
        h, w = sample_frame.shape[:2]
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        # Запись 5 секунд при 15 FPS = ~75 кадров
        for _ in range(75):
            r, f = stream_reader.read()
            if r and f is not None:
                out.write(f)
            asyncio.run(asyncio.sleep(1 / fps))
            
        out.release()
        return True

    success = await asyncio.to_thread(record_clip)

    if not success or not os.path.exists(video_path):
        await msg.edit_text("❌ Не удалось записать видеофрагмент.")
        return

    video_file = FSInputFile(video_path)
    await message.reply_video(video=video_file, caption="🎥 Видеофрагмент 5 сек")
    await msg.delete()
    
    if os.path.exists(video_path):
        os.remove(video_path)


@router.message(Command("relay"))
@router.message(lambda m: m.text == "⚡ Свет/Реле")
async def cmd_relay_toggle(message: types.Message):
    """Переключение реле Sonoff через eWeLink API/LAN."""
    try:
        relay = SonoffRelay()
        await relay.turn_on()
        await message.reply("⚡ Команда отправлена на Sonoff: реле включено.")
    except Exception as e:
        await message.reply(f"❌ Ошибка eWeLink: {e}")
         
@router.message(Command("status"))
@router.message(lambda m: m.text == "📊 Статус сервера")
async def cmd_status(message: types.Message):
    """Показывает загрузку ЦП, ОЗУ и диска на системном блоке."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    text = (
        f"🖥 **Статус Ubuntu Server:**\n\n"
        f"⚙️ **Загрузка ЦП:** {cpu}%\n"
        f"🧠 **ОЗУ:** {ram.percent}% ({ram.used // (1024**2)} МБ / {ram.total // (1024**2)} МБ)\n"
        f"💾 **Диск:** {disk.percent}% свободно {disk.free // (1024**3)} ГБ"
    )
    await message.reply(text, parse_mode="Markdown")


@router.message(Command("testurl"))
async def cmd_testurl(message: types.Message):
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
        await message.reply(f"❌ Не удалось получить кадр с '{name}' за отведённое время.")
        return

    h, w = frame.shape[:2]
    jpeg_bytes = await asyncio.to_thread(encode_jpeg, frame)
    input_file = BufferedInputFile(jpeg_bytes, filename=f"{name}.jpg")
    await message.reply_photo(photo=input_file, caption=f"📷 {name}: {w}x{h}\n{url}")

