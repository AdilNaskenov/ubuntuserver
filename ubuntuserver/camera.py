"""
Всё, что связано с получением кадров с камеры: постоянный низкоразрешённый
поток (для движения/YOLO) и разовые высокоразрешённые снимки по требованию.
"""
import time
import threading
import logging
import cv2

import config

logger = logging.getLogger(__name__)


class RTSPStreamReader:
    """
    Отдельный поток (thread), который в фоне постоянно держит соединение
    открытым и хранит в self.frame самый свежий кадр. Основной код просто
    забирает что там есть в любой момент через .read(), не дожидаясь камеру.
    """

    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap.open(self.src)
                continue

            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = True
            else:
                self.ret = False
                time.sleep(0.02)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()


def grab_single_frame(stream_url, timeout_sec=8):
    """
    Открывает RTSP-соединение, забирает ОДИН кадр и сразу закрывает —
    в отличие от RTSPStreamReader, не держит соединение постоянно открытым.
    Используется для разовых высокоразрешённых снимков (stream1) и для
    команды /testurl, где не нужен непрерывный поток.

    Первые несколько кадров у некоторых камер сразу после подключения бывают
    пустыми/чёрными — поэтому пробуем читать в цикле в пределах таймаута,
    а не полагаемся на самый первый read().
    """
    cap = cv2.VideoCapture(stream_url)
    start = time.time()
    frame = None
    try:
        if cap.isOpened():
            while time.time() - start < timeout_sec:
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
                    break
                time.sleep(0.1)
    finally:
        cap.release()
    return frame


def grab_high_res_frame():
    """Тонкая обёртка над grab_single_frame с настройками по умолчанию под
    высокоразрешённый поток (stream1). Вызывается синхронно, поэтому должна
    запускаться через asyncio.to_thread() из основного цикла."""
    return grab_single_frame(config.HIGH_RES_STREAM_URL, config.HIGH_RES_GRAB_TIMEOUT)


# --- Текущий активный поток с камеры ---
# Заполняется в main.py при старте video_processing_loop(), нужен, чтобы
# команда /photo могла взять самый свежий кадр без создания нового
# подключения к камере (используется уже открытый поток).
#
# ВАЖНО: доступ через функции ниже, а не через `from camera import
# current_stream_reader` — при импорте по имени Python привязывает
# переменную к значению НА МОМЕНТ ИМПОРТА (None) и не увидит обновление
# после set_current_stream_reader(). Через функции всегда читается
# актуальное значение.
current_stream_reader = None


def set_current_stream_reader(reader):
    global current_stream_reader
    current_stream_reader = reader


def get_current_stream_reader():
    return current_stream_reader
