import tensorflow as tf
from tensorflow import keras
from keras import layers


def train_input_fn(path='data'):
    """
    Create and return TF function for model training

    Note: The way the data is setup:

     - 0 is closed
     - 1 is open
    """
    def fn():
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            label_mode='binary',
            batch_size=1
        )
        ds = ds.map(lambda x, y: ({"x": x}, y))

        return ds.repeat()
    return fn


def get_training_data(path='data/train'):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            label_mode='binary',
            batch_size=1,
            image_size=(256, 256),
            color_mode="grayscale",
        )
    return ds

def get_model(model_dir=None, warm_start=False):
    """
    Return basic linear classifier
    """
    model = keras.models.Sequential([
        layers.Input((256, 256, 1)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(8, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(
        optimizer='sgd',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def main():
    model = get_model()
    ds = get_training_data()
    model.fit(ds, epochs=20)
    model.save('model.keras')


if __name__ == '__main__':
    main()
