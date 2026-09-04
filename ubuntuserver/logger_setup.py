"""
Общая настройка логирования. Импортируется первым делом в main.py, чтобы
basicConfig() был вызван ровно один раз до того, как остальные модули
начнут писать в лог через logging.getLogger(__name__).
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
