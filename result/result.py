import time
import cv2
import keyboard
import mss
import numpy as np
import tensorflow as tf

# 2. ЗАГРУЗКА МОДЕЛИ
model_path = "dino.keras"
model = tf.keras.models.load_model(model_path)

# размеры
img_height = 128
img_width = 128

# Порог уверенности
jump_threshold = 0.85

with mss.mss() as sct:
    monitor = {"width": 350, "height": 185, "left": 480, "top": 220} # изменил координаты, так как сохраненные координаты фото не совпадают с mss

    while True:
        if keyboard.is_pressed("q"):
            print("Бот остановлен.")
            break

        screen = sct.grab(monitor)
        screen_array = np.array(screen)[:, :, :3]  # Срез до BGR

        resized_frame = cv2.resize(screen_array, (img_width, img_height))

        # Переводим в float32 для слоя Rescaling
        ready_frame = resized_frame.astype(np.float32)
        input_data = np.expand_dims(ready_frame, axis=0)

        # Предсказания
        prediction = float(model(input_data, training=False).numpy())

        # 5. МГНОВЕННЫЙ КЛИК БЕЗ ЗАЖАТИЯ
        if prediction > jump_threshold:
            # Стреляем кликом мгновенно
            keyboard.press("space")
            keyboard.release("space")

            # пауза, чтобы бот не спамил пробелом, пока он в воздухе
            time.sleep(0.18)

        # время для стабильности Windows-потока
        time.sleep(0.002)
