"""
Всё, что связано с интерактивной Re-ID разметкой в Telegram: клавиатура
выбора особи, отправка кропа на разметку (с AI-подсказкой для коров),
обработка нажатий на кнопки и ответа с кличкой новой особи.
"""
import logging
import cv2
from aiogram import Router, types
from aiogram.types import BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
import asyncio

import config
from bot_instance import bot
from reid_classifier import cow_reid_classifier
import animals_store as store

logger = logging.getLogger(__name__)
router = Router()


def build_reid_keyboard(class_name, message_id, suggested_animal_id=None):
    """
    suggested_animal_id — если задан, кнопка этой особи ставится ПЕРВОЙ
    и помечается 🤖, чтобы человеку было видно, что это предложение ИИ,
    а не просто первая в списке по случайному порядку.
    """
    builder = InlineKeyboardBuilder()
    animals = store.animals_db.get(class_name, {})

    # Сортируем так, чтобы предложенная ИИ особь оказалась первой в списке —
    # меньше скроллить/искать глазами, если подсказка верна.
    animal_items = list(animals.items())
    if suggested_animal_id is not None:
        animal_items.sort(key=lambda item: item[0] != suggested_animal_id)

    for animal_id, name in animal_items:
        label = f"🤖 {name}" if animal_id == suggested_animal_id else name
        builder.button(text=label, callback_data=f"reid|{class_name}|{animal_id}|{message_id}")

    builder.button(text="➕ Новая особь", callback_data=f"reid|{class_name}|new|{message_id}")
    builder.button(text="❌ Не тот класс", callback_data=f"reid|{class_name}|wrongclass|{message_id}")
    builder.adjust(2)
    return builder.as_markup()


async def send_crop_for_labeling(crop_path, class_name, full_frame_path):
    """
    Перед отправкой на ручную разметку — если это корова и Re-ID классификатор
    загружен — прогоняем кроп через модель и добавляем подсказку в подпись
    и в клавиатуру. Для "horse" подсказки не будет (модель обучена только на
    коровах) — просто обычная ручная разметка, как раньше.
    """
    suggested_animal_id = None
    suggestion_caption = ""

    if class_name == "cow" and cow_reid_classifier is not None:
        try:
            crop_img = cv2.imread(crop_path)
            if crop_img is not None:
                animal_id, folder_name, confidence = await asyncio.to_thread(
                    cow_reid_classifier.predict, crop_img
                )
                if animal_id is not None and confidence >= config.REID_MIN_CONFIDENCE_TO_SHOW:
                    # Каноничное имя берём из animals_db (могло отличаться от
                    # имени папки, если то "очищалось" от спецсимволов при
                    # сохранении) — если особи уже нет в базе, покажем как есть.
                    display_name = store.animals_db.get("cow", {}).get(animal_id, folder_name)
                    suggested_animal_id = animal_id
                    suggestion_caption = f"\n🤖 Похоже на: {display_name} ({confidence:.0%})"
        except Exception as e:
            logger.error(f"Ошибка Re-ID предсказания: {e}")

    try:
        with open(crop_path, "rb") as f:
            photo_bytes = f.read()
        input_file = BufferedInputFile(photo_bytes, filename=crop_path.split("/")[-1])
        msg = await bot.send_photo(
            chat_id=config.REID_CHAT_ID,
            photo=input_file,
            caption=f"Кто это? ({class_name}){suggestion_caption}"
        )
        store.pending_crops[msg.message_id] = (class_name, crop_path, full_frame_path)
        keyboard = build_reid_keyboard(class_name, msg.message_id, suggested_animal_id)
        await bot.edit_message_reply_markup(
            chat_id=config.REID_CHAT_ID, message_id=msg.message_id, reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Ошибка отправки кропа на Re-ID разметку: {e}")


@router.callback_query(lambda c: c.data and c.data.startswith("reid|"))
async def handle_reid_callback(callback: types.CallbackQuery):
    _, class_name, animal_id, message_id_str = callback.data.split("|")
    message_id = int(message_id_str)

    if message_id not in store.pending_crops:
        await callback.answer("Этот кроп уже обработан или устарел.", show_alert=True)
        return

    crop_class, crop_path, full_frame_path = store.pending_crops[message_id]

    if animal_id == "new":
        store.waiting_for_name[callback.message.chat.id] = (crop_class, message_id)
        await callback.message.reply("Напиши кличку/номер новой особи ответом на это сообщение.")
        await callback.answer()
        return

    if animal_id == "wrongclass":
        saved_frame_path = await store.save_crop_as_misclassified(crop_class, crop_path, full_frame_path)
        del store.pending_crops[message_id]

        await callback.message.edit_caption(caption=f"❌ Класс неверный ({crop_class}) — на переразметку")

        with open(saved_frame_path, "rb") as f:
            frame_bytes = f.read()
        frame_file = BufferedInputFile(frame_bytes, filename=saved_frame_path.split("/")[-1])
        await callback.message.reply_photo(
            photo=frame_file,
            caption=f"Полный кадр для переразметки (был определён как {crop_class}):\n{saved_frame_path}"
        )

        await callback.answer("Отмечено как неверный класс")
        return

    animal_name = store.animals_db.get(crop_class, {}).get(animal_id, animal_id)
    await store.save_crop_to_reid(crop_class, animal_id, animal_name, crop_path)
    await store.discard_pending_full_frame(full_frame_path)
    del store.pending_crops[message_id]

    await callback.message.edit_caption(caption=f"✅ {crop_class}: {animal_name}")
    await callback.answer(f"Сохранено: {animal_name}")


@router.message()
async def handle_name_reply(message: types.Message):
    """
    Ловит ответ с кличкой новой особи (Re-ID). ВАЖНО: этот хендлер без фильтров
    должен быть зарегистрирован ПОСЛЕДНИМ из всех хендлеров сообщений во всём
    боте — то есть reid_router должен подключаться в main.py ПОСЛЕ
    commands_router. aiogram проверяет роутеры/хендлеры в порядке подключения
    и использует первый подошедший, так что если бы этот хендлер без фильтров
    шёл раньше — он перехватывал бы вообще любой текст на себя, включая
    команды, и остальные обработчики просто не получили бы управление.
    """
    chat_id = message.chat.id
    if chat_id not in store.waiting_for_name or not message.text:
        return

    class_name, message_id = store.waiting_for_name.pop(chat_id)
    if message_id not in store.pending_crops:
        return

    crop_class, crop_path, full_frame_path = store.pending_crops[message_id]
    store.animals_db.setdefault(crop_class, {})
    animal_name = message.text.strip()

    existing_id = next(
        (aid for aid, name in store.animals_db[crop_class].items()
         if name.strip().lower() == animal_name.lower()),
        None
    )

    if existing_id is not None:
        final_id = existing_id
        is_new = False
        folder_name = store.animals_db[crop_class][existing_id]
    else:
        existing_ids = [int(i) for i in store.animals_db[crop_class].keys()] if store.animals_db[crop_class] else []
        final_id = str(max(existing_ids, default=0) + 1)
        store.animals_db[crop_class][final_id] = animal_name
        store.save_animals(store.animals_db)
        is_new = True
        folder_name = animal_name

    await store.save_crop_to_reid(crop_class, final_id, folder_name, crop_path)
    await store.discard_pending_full_frame(full_frame_path)
    del store.pending_crops[message_id]

    if is_new:
        await bot.send_message(chat_id, f"✅ Новая особь добавлена: {animal_name} ({crop_class})")
    else:
        await bot.send_message(chat_id, f"✅ Добавлено к уже существующей особи: {animal_name} ({crop_class})")
