from tensorflow.keras import models, layers
from config import IMG_SIZE, MODEL_TYPE
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB3,
    DenseNet121,
    ResNet50
)

def build_model(fine_tune=False):
    # ===============================
    # ✅ اختيار الموديل من config.py
    # ===============================
    if MODEL_TYPE == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif MODEL_TYPE == "EfficientNetB3":
        base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif MODEL_TYPE == "DenseNet121":
        base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif MODEL_TYPE == "ResNet50":
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    else:
        raise ValueError(f"❌ Unknown model type: {MODEL_TYPE}")

    # ===============================
    # Fine-tuning configuration
    # ===============================
    if not fine_tune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # فتح آخر 100 طبقة فقط
        for layer in base_model.layers[:-100]:
            layer.trainable = False

    # ===============================
    # بناء الموديل النهائي
    # ===============================
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
