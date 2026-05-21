# ===============================
# ⚙️ Config for BraTS2020 Segmentation
# ===============================

import os

# 📁 المسارات الأساسية
DATA_DIR = r"D:\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
MODEL_DIR = r"D:\archive\BraTS2020_model_seg"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🧩 إعدادات البيانات
MODALITY = "flair"  # ممكن تغيرها لاحقًا لـ "t1ce" أو "t2" أو حتى دمجهم
PATCH_SIZE = (96, 96, 96)  # حجم الباتش اللي هنقصّه من الصور 3D

# ⚙️ إعدادات التدريب
BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 1e-4
SEED = 42

# 💾 إعداد حفظ النموذج
MODEL_NAME = "unet3d_brats2020.pth"


