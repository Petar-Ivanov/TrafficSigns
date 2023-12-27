import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    valid_input = False

    while valid_input == False:

        choice = input("Опции:\n1. Обучаване на модел\n2. Използване на модел\n")

        if choice == "1":
            valid_input = True
            data_dir = input("Директория на данните: ")
            model_dir = input("Директория на модела: ")

            # Зареждане на тренировъчните данни
            images, labels = load_data(data_dir)

            # Разделяне на данните
            labels = tf.keras.utils.to_categorical(labels)
            x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=TEST_SIZE)

            # Създаване на модел
            model = get_model()

            # Обучение на модела
            model.fit(x_train, y_train, epochs=EPOCHS)
            
            # Оценка на модела
            model.evaluate(x_test,  y_test, verbose=2)

            # Запаметяване на модела
            model.save(model_dir)
            print(f"Model saved to {model_dir}.")

        elif choice == "2":
            valid_input = True
            model_dir = input("Директория на модела: ")
            img_dir = input("Директория на изображението: ")

            # Зареждане на модела
            loaded_model = tf.keras.models.load_model(model_dir)

            # Зареждане и преоразмеряване на изображението
            img = cv2.imread(img_dir)
            res_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            input_image = np.expand_dims(res_img, axis=0)

            # Правене на предположение
            predictions = loaded_model.predict(input_image)
            predicted_category = np.argmax(predictions)

            print(f"Предполагаемата категория е: {predicted_category}")

        else:
            print("Валидни опции: 1/2")




def load_data(data_dir):
    """
    Зареждане на данните от посочената директория 'data_dir'.

    За всяка категория е създадена отделна поддиректория в 'data_dir', чието
    име е името на категорията. Те започват от 0 до NUM_CATEGORIES - 1.
    Във всяка поддиректория има множество изображения.

    След извличане на данните от файловата система връщаме tuple (images, labels),
    където images е списък с всички изображения, които ще се използват за обучението
    на невронната мрежа, а labels е списък с отговарящите им етикети (името на
    поддиректорията, от която са извлечени).

    Изобраченията идват в различни размери, затова ги преоразмеряваме според константите
    IMG_WIDTH и IMG_HEIGHT.
    """
    images = []
    labels = []

    for folder in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, folder)):
            img = cv2.imread(os.path.join(data_dir, folder, file))
            res_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(res_img)
            labels.append(folder)

    return (images, labels)

def get_model():
    """
    Създаване на последователен Deep Convolutional Neural Network модел с два
    конволюционни слоя, две прилагания на MaxPooling, два скрити слоя и 
    използване на Dropout техника за избягване на overfitting.
    """
    # Създаване на последователен модел чрез TensorFlow Keras
    model = tf.keras.models.Sequential([
        # Конволюционен слой с 32 филтъра и kernel 3x3 
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_WIDTH, 3)),

        # Max-Pooling с цел смачкване на изображението за по-лесно трениране на модела
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Втори конволюционен слой с 32 филтъра и kernel 3x3 
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_WIDTH, 3)),

        # Втори Max-Pooling с цел смачкване на изображението за по-лесно трениране на модела
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Flattening с цел представяне на данните за отделните пиксели като едноизмерна структура от данни (Input Layer)
        tf.keras.layers.Flatten(),

        # Hidden layer с 172 неврона
        tf.keras.layers.Dense(172, activation="relu"),

        # Втори Hidden layer с 86 неврона
        tf.keras.layers.Dense(86, activation="relu"),

        # Прилагане на Dropout техника за избягване на overfitting, при която при всяка тренировка на мрежата изкючваме случайни 30% от невроните 
        tf.keras.layers.Dropout(0.3),

        # Output layer с по 1 неврон за всяка една от възможните категории
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Компилиране на модела
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
