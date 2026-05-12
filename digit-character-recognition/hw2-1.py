import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import time


def load_emnist():
    print("Loading EMNIST dataset...")
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/balanced',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.transpose(image, [1, 0, 2])
        return image, label

    batch_size = 128
    ds_train = ds_train.map(normalize_img).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def build_model(use_pooling=True, conv_stride=(1, 1), padding_type='valid'):
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type))

    if use_pooling:
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.Conv2D(32, (3, 3), strides=conv_stride, activation='relu', padding=padding_type))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type))

    if use_pooling:
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.Conv2D(64, (3, 3), strides=conv_stride, activation='relu', padding=padding_type))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(47, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


ds_train, ds_test = load_emnist()

configs = [
    {"name": "Baseline (Max Pooling)", "use_pooling": True, "stride": (1, 1), "pad": "valid"},
    {"name": "No Pooling, Stride (1,2)", "use_pooling": False, "stride": (1, 2), "pad": "valid"},
    {"name": "No Pooling, Stride (2,1)", "use_pooling": False, "stride": (2, 1), "pad": "valid"},
    {"name": "No Pooling, Stride (2,2)", "use_pooling": False, "stride": (2, 2), "pad": "valid"},
    {"name": "No Pooling, Stride (2,2) + Same Padding", "use_pooling": False, "stride": (2, 2), "pad": "same"},
]

epochs = 5

for config in configs:
    print(f"\n--- Training Model: {config['name']} ---")
    model = build_model(use_pooling=config['use_pooling'],
                        conv_stride=config['stride'],
                        padding_type=config['pad'])

    start_time = time.time()
    history = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
    end_time = time.time()

    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    print(f"Result for {config['name']}:")
    print(f"Accuracy: {test_acc:.4f} | Training Time: {end_time - start_time:.2f} seconds")
