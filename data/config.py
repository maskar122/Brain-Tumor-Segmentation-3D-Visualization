# ===============================
# 🧠 BraTS2020 Classification Config
# ===============================


# مسارات البيانات والموديل
DATA_DIR = r"D:\archive\BraTS2020_2D_split"
MODEL_DIR = r"D:\archive\BraTS2020_model"

# إعدادات الصور والموديل
IMG_SIZE = (300, 300)        # لو EfficientNetB0 خليه (224,224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# اختيار الموديل (اكتب اسم واحد فقط من القائمة):
# "EfficientNetB0" / "EfficientNetB3" / "DenseNet121" / "ResNet50"
MODEL_TYPE = "EfficientNetB3"

# اسم الملف عند الحفظ (صيغة h5)
MODEL_NAME = "brats2020_classifier_final.h5"
