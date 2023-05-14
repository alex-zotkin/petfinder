import tensorflow as tf
import os
import keras.utils as image
import numpy as np

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



img_width = img_height = 150

# predicting images
img = image.load_img('test/test2.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

for mark in marks:
    print(mark)
    model = tf.keras.models.load_model("models/" + mark + ".h5")

    classes = model.predict(images, batch_size=10)
    print("[", classes[0][0], ",", classes[0][1], "]")
    print("NO" if classes[0][0] > classes[0][1] else "YES")
