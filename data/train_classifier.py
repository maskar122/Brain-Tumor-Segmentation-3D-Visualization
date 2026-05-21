import os
from config import MODEL_DIR, EPOCHS, MODEL_NAME
from data_loader import load_datasets
from model_builder import build_model
from callbacks import get_callbacks

# تحميل البيانات
train_ds, val_ds, test_ds = load_datasets()

# ========== المرحلة الأولى: تدريب مبدئي ==========
model = build_model(fine_tune=False)
print("✅ Training Base Model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks()
)

# تقييم مبدئي
loss, acc = model.evaluate(test_ds)
print(f"Base Accuracy: {acc*100:.2f}%")

# ========== المرحلة الثانية: Fine-Tuning ==========
print("\n🎯 Fine-Tuning Phase...")
fine_tune_model = build_model(fine_tune=True)
fine_tune_model.set_weights(model.get_weights())

fine_tune_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=get_callbacks()
)

loss, acc = fine_tune_model.evaluate(test_ds)
print(f"🎯 Final Fine-Tuned Accuracy: {acc*100:.2f}%")

# حفظ الموديل بصيغة .h5
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
fine_tune_model.save(model_path)
print(f"✅ Model saved to: {model_path}")
