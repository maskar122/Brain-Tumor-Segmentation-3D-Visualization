# ===============================
# 🧠 3D Visualization of Predicted Tumor Segmentation (Auto Path Detection)
# ===============================

import os
import glob
import nibabel as nib
import numpy as np
import pyvista as pv

# ===============================
# ⚙️ إعدادات المريض والملفات
# ===============================
PATIENT_ID = "BraTS20_Training_356"  # ← غيّر الاسم حسب المريض اللي عملت له segmentation
DATA_DIR = r"D:\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# 🔍 نبحث تلقائيًا عن ملفات Flair و Segmentation في كل المجلدات
flair_candidates = glob.glob(os.path.join(DATA_DIR, "**", f"{PATIENT_ID}_flair.nii"), recursive=True)
mask_candidates = glob.glob(os.path.join(DATA_DIR, "**", f"{PATIENT_ID}_predicted_seg.nii"), recursive=True)

if not flair_candidates:
    raise FileNotFoundError(f"❌ Flair file not found for {PATIENT_ID} inside {DATA_DIR}")
if not mask_candidates:
    raise FileNotFoundError(f"❌ Predicted segmentation not found for {PATIENT_ID} inside {DATA_DIR}")

flair_path = flair_candidates[0]
mask_path = mask_candidates[0]

print(f"✅ Using FLAIR: {flair_path}")
print(f"✅ Using predicted mask: {mask_path}")

# ===============================
# 📥 تحميل البيانات
# ===============================
flair = nib.load(flair_path).get_fdata().astype(np.float32)
mask = nib.load(mask_path).get_fdata().astype(np.float32)

# Normalize flair intensity
flair = np.clip(flair, 0, 2000)
flair = flair / np.max(flair)

# ===============================
# 🎨 إعداد PyVista
# ===============================
p = pv.Plotter()
p.add_volume(pv.wrap(flair), cmap="gray", opacity=0.15, name="Brain")

# ===============================
# 🧩 تعريف طبقات الورم
# ===============================
tumor_labels = {
    1: ("Necrotic Core", "red"),
    2: ("Edema", "green"),
    3: ("Enhancing Tumor", "yellow"),
}

# ===============================
# 🧱 رسم المناطق بالـ 3D
# ===============================
for label, (name, color) in tumor_labels.items():
    tumor_mask = np.where(mask == label, 1, 0)
    voxel_count = np.sum(tumor_mask)
    if voxel_count > 200:  # تجاهل الضوضاء الصغيرة
        tumor_surface = pv.wrap(tumor_mask).contour(isosurfaces=[0.5])
        p.add_mesh(tumor_surface, color=color, opacity=0.4, name=name)
        print(f"✅ Added region: {name} ({color}) voxels={voxel_count}")
    else:
        print(f"⚠️ Skipped {name} ({color}) — too few voxels ({voxel_count})")

# ===============================
# 🧭 إعداد العرض النهائي
# ===============================
p.add_axes()
p.show_bounds(grid='front', location='outer', all_edges=True)
p.show(title=f"3D Tumor Segmentation - {PATIENT_ID}")
