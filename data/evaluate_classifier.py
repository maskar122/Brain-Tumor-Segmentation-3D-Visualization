# ===============================
# 🎯 Evaluation Script for BraTS2020 Classifier
# ===============================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score
)

from config import MODEL_DIR, MODEL_NAME
from data_loader import load_datasets

# -------------------------------
# 1️⃣ تحميل البيانات والموديل
# -------------------------------
_, _, test_ds = load_datasets()

model_path = os.path.join(MODEL_DIR, MODEL_NAME)
print(f"🔍 Loading model from: {model_path}")

model = tf.keras.models.load_model(model_path)

# -------------------------------
# 2️⃣ حساب التنبؤات
# -------------------------------
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# -------------------------------
# 3️⃣ تقييم الأداء العددي
# -------------------------------
acc = accuracy_score(y_true, y_pred_labels)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

print("📋 Classification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=['LGG', 'HGG']))

# -------------------------------
# 4️⃣ رسم الـ Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 5️⃣ رسم ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
