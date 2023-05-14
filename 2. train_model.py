from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

marks = [
    "eyes",
    "face",
    "near",
    "accessory",
    "group",
    "collage",
    "human",
    "occlusion",
    "info",
    "blur",
]

for mark in marks:
    print(mark.upper())

    base_dir = os.path.dirname(__file__)
    train_dir = os.path.join(base_dir, 'train', mark)
    test_dir = os.path.join(base_dir, 'test', mark)

    train_mark0_dir = os.path.join(train_dir, '0')
    train_mark1_dir = os.path.join(train_dir, '1')
    test_mark0_dir = os.path.join(test_dir, '0')
    test_mark1_dir = os.path.join(test_dir, '1')

    num_mark0_tr = len(os.listdir(train_mark0_dir))
    num_mark1_tr = len(os.listdir(train_mark1_dir))
    num_mark0_test = len(os.listdir(test_mark0_dir))
    num_mark1_test = len(os.listdir(test_mark1_dir))

    total_train = num_mark0_tr + num_mark1_tr
    total_test = num_mark0_test + num_mark1_test


    BATCH_SIZE = 100 # количество тренировочных изображений для обработки перед обновлением параметров модели
    IMG_SHAPE = 150 # размерность к которой будет преведено входное изображение
    train_image_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=45,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest'
                                               )
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=train_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')
    test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=test_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMG_SHAPE, IMG_SHAPE),
                                                                  class_mode='binary')




    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    EPOCHS = 200
    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=test_data_gen,
        validation_steps=int(np.ceil(total_test / float(BATCH_SIZE)))
    )

    model.save("models/" + mark + ".h5")
