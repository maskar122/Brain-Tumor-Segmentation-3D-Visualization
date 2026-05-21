import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED

def load_datasets():
    # تحميل الصور
    train_ds = image_dataset_from_directory(
        DATA_DIR + "\\train",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR + "\\val",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = image_dataset_from_directory(
        DATA_DIR + "\\test",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # تحويل الصور من grayscale إلى RGB
    def to_rgb(x, y):
        x = tf.image.grayscale_to_rgb(x)
        return x, y

    # Augmentation قوية
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])

    # تجهيز الـdatasets
    train_ds = train_ds.map(to_rgb).map(
        lambda x, y: (data_augmentation(x, training=True), y)
    ).cache().shuffle(500).prefetch(AUTOTUNE)

    val_ds = val_ds.map(to_rgb).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(to_rgb).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
