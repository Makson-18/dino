# импорт библиотек
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# путь к датасету
base_dir = r"C:\Users\RobotComp.ru\Desktop\flapyy\linear"

batch_size = 32
img_height = 128
img_width = 128

# 2. тренировочные данны 80 процентов; 20 валидационные
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Оптимизация памяти
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Архитектура модели
model = models.Sequential([
    layers.Rescaling(1 / 255.0, input_shape=(img_height, img_width, 3)), #нормализация

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# Сборка модели
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#  обучение
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# Сохранение
model.save("dino.keras")

# График
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.title("График обучения")

plt.plot(loss, label="Ошибка на тренировочных данных")
plt.plot(val_loss, label="Ошибка на валидационных данных")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.show()
