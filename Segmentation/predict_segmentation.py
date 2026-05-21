# ===============================
# 🎯 Generate Segmentation Prediction for One Patient (Fixed Version)
# ===============================

import os
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# ===============================
# ⚙️ إعدادات المسارات
# ===============================
DATA_DIR = r"D:\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
MODEL_PATH = r"D:\archive\BraTS2020_model_seg\unet3d_epoch2_dice0.861.pth"
PATIENT_ID = "BraTS20_Training_001"   # ← غيّر الاسم لو حابب مريض آخر

PATIENT_DIR = os.path.join(DATA_DIR, PATIENT_ID)
OUTPUT_PATH = os.path.join(PATIENT_DIR, f"{PATIENT_ID}_predicted_seg.nii")

# ===============================
# 🧩 تحميل الموديل (بنفس الإعدادات اللي استخدمناها في التدريب)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,     # ✅ لازم نحافظ على نفس عدد الوحدات
    norm='batch'         # ✅ نفس إعدادات التطبيع المستخدمة أثناء التدريب
).to(device)

print(f"🔍 Loading model from: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===============================
# 📥 تحميل صورة Flair MRI للمريض
# ===============================
flair_path = os.path.join(PATIENT_DIR, f"{PATIENT_ID}_flair.nii")
if not os.path.exists(flair_path):
    raise FileNotFoundError(f"❌ FLAIR file not found: {flair_path}")

print(f"✅ Found FLAIR MRI for patient: {PATIENT_ID}")

flair = nib.load(flair_path).get_fdata().astype(np.float32)
flair = np.clip(flair, 0, 2000)
flair = flair / 2000.0

# تحويل إلى Tensor
input_tensor = torch.from_numpy(flair).unsqueeze(0).unsqueeze(0).to(device)

# ===============================
# 🔮 تطبيق الموديل للتنبؤ
# ===============================
print(f"🧠 Running segmentation inference for: {PATIENT_ID} ...")

with torch.no_grad():
    output = sliding_window_inference(input_tensor, (128, 128, 128), 1, model)
    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

# ===============================
# 💾 حفظ الناتج كـ NIfTI
# ===============================
nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine=np.eye(4)), OUTPUT_PATH)
print(f"✅ Segmentation saved to: {OUTPUT_PATH}")

# ===============================
# 🔍 التحقق من القيم داخل الماسك
# ===============================
unique_labels = np.unique(prediction)
print(f"🎯 Unique labels in predicted mask: {unique_labels}")
print("✅ Done! Segmentation inference completed successfully.")
