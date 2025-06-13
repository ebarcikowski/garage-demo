"""Test model with CLI script."""
import logging
logging.getLogger('tensorflow').disabled = True

from tensorflow import keras
import tensorflow as tf
import numpy as np
from glob import glob


DATA_PATH = 'data/test'

CLASSES = [
    "closed",
    "open"
]


def load_image(path):
    img = keras.utils.load_img(
        path,
        color_mode="grayscale",
        target_size=(256, 256)
    )
    img_array = keras.utils.img_to_array(img)

    return img_array

def main():
    model = keras.models.load_model('model.keras')

    for img_fn in glob(DATA_PATH + '/*/*'):
        img_array = load_image(img_fn)
        img_array = tf.expand_dims(img_array, 0)
        predicts = model.predict(img_array, verbose=0)
        idx = np.argmax(predicts)
        print(f"{img_fn}: {CLASSES[idx]}")

if __name__ == '__main__':
    main()
