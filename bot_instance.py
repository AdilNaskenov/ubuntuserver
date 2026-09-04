"""
Создаёт единственный экземпляр Bot на весь проект. Другие модули делают
`from bot_instance import bot`, когда им нужно явно вызвать bot.send_photo()
и т.п. (объект message в хендлерах aiogram уже сам умеет message.reply()/
message.answer() без импорта bot — он использует привязанный к себе бот
автоматически).

Dispatcher создаётся отдельно в main.py, не здесь — чтобы не создавать
циклических импортов между модулями с роутерами.
"""
from aiogram import Bot
import config

bot = Bot(token=config.BOT_TOKEN)
