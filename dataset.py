# импорт библиотек
import os
import time
import cv2
import keyboard
import mss
import numpy as np

# путь к папка
file_no = r"C:\Users\RobotComp.ru\Desktop\flapyy\linear\no"
file_yes = r"C:\Users\RobotComp.ru\Desktop\flapyy\linear\yes"

count_no = 0
count_yes = 0
skip_counter = 0

with mss.mss() as sct:
    monitor = {"width": 900, "height": 223, "top": 160, "left": 515} # разрешение для игры

    while True:
        if keyboard.is_pressed("q"): # выход по кнопки
            print(f"Всего фото: YES = {count_yes}, NO = {count_no}")
            break

        # Делаем скриншот текущего кадра
        screen = sct.grab(monitor)
        screen_array = np.array(screen)[:, :, :3]

        # сохраняем в папку для прыжков
        if keyboard.is_pressed("space"):
            count_yes += 1
            file_name = f"image_{count_yes}.png"
            path = os.path.join(file_yes, file_name)
            cv2.imwrite(path, screen_array)

            while keyboard.is_pressed("space"): # чтобы не было спама при нажати на пробел
                time.sleep(0.01)

        else:
            # сохраняем в папку для бездействия
            skip_counter += 1
            if skip_counter % 7 == 0:
                count_no += 1
                file_name = f"image_{count_no}.png"
                path = os.path.join(file_no, file_name)

                cv2.imwrite(path, screen_array)

        # Микро-задержка для стабильности
        time.sleep(0.07)
