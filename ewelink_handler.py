import os
import logging
from ewelink import ewelink
from dotenv import load_dotenv
# Загружаем данные из .env
load_dotenv()

class SonoffRelay:
    def __init__(self):
        self.email = os.getenv("EWELINK_EMAIL", "adil.n2006@icloud.com")
        self.password = os.getenv("EWELINK_PASSWORD", "Karaganda23")
        self.region = os.getenv("EWELINK_REGION", "eu")
        self.device_id = os.getenv("DEVICE_ID", "10025a6a13")
        
        # Инициализируем клиент
        self.client = Ewelink(
            email=self.email,
            password=self.password,
            region=self.region
        )

    async def turn_on(self) -> bool:
        """Включить реле"""
        try:
            await self.client.set_device_power_state(self.device_id, "on")
            return True
        except Exception as e:
            logging.error(f"Ошибка включения реле: {e}")
            return False

    async def turn_off(self) -> bool:
        """Выключить реле"""
        try:
            await self.client.set_device_power_state(self.device_id, "off")
            return True
        except Exception as e:
            logging.error(f"Ошибка выключения реле: {e}")
            return False

    async def toggle(self) -> bool:
        """Переключить состояние (Вкл -> Выкл / Выкл -> Вкл)"""
        try:
            await self.client.toggle_device(self.device_id)
            return True
        except Exception as e:
            logging.error(f"Ошибка переключения реле: {e}")
            return False

    async def get_status(self) -> str:
        """Получить текущее состояние (on / off / offline)"""
        try:
            device_info = await self.client.get_device(self.device_id)
            if not device_info.get("online", False):
                return "offline"
            return device_info.get("params", {}).get("switch", "unknown")
        except Exception as e:
            logging.error(f"Ошибка получения статуса: {e}")
            return "error"
