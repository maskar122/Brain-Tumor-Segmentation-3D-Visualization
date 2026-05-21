# ===============================
# 📊 Evaluation + Clinical Linking for BraTS2020 3D Segmentation
# ===============================

import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from config_seg import DATA_DIR, MODEL_DIR, MODEL_NAME, PATCH_SIZE, MODALITY
from model_seg import get_unet3d

# ========================================
# ⚙️ إعداد المسارات
# ========================================
CSV_DIR = r"D:\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
NAME_MAP_PATH = os.path.join(CSV_DIR, "name_mapping.csv")
SURV_PATH = os.path.join(CSV_DIR, "survival_info.csv")

VIS_DIR = os.path.join(MODEL_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔎 Using device for evaluation: {device}")

# ========================================
# 🏆 اختيار أفضل موديل
# ========================================
def pick_best_checkpoint(model_dir, default_name):
    pattern = re.compile(r"unet3d_epoch(\d+)_dice([0-9.]+)\.pth$")
    best_path, best_dice = None, -1.0
    for p in glob.glob(os.path.join(model_dir, "unet3d_epoch*_dice*.pth")):
        m = pattern.search(os.path.basename(p))
        if m:
            dice_val = float(m.group(2))
            if dice_val > best_dice:
                best_dice, best_path = dice_val, p
    if best_path:
        print(f"🏆 Best checkpoint found: {os.path.basename(best_path)} (dice={best_dice:.4f})")
        return best_path
    # fallback
    fallback = os.path.join(model_dir, default_name)
    print(f"ℹ️ No epoch-dice file found, fallback to: {fallback}")
    return fallback

ckpt_path = pick_best_checkpoint(MODEL_DIR, MODEL_NAME)
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

# ========================================
# 🧠 تحميل الموديل
# ========================================
model = get_unet3d().to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()
print("✅ Model weights loaded successfully.\n")

# ========================================
# 🧩 تجهيز بيانات الـ Validation
# ========================================
def build_val_file_lists(data_dir, modality):
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    images, labels, patient_ids = [], [], []
    for subject in subject_dirs:
        name = os.path.basename(subject)
        img = os.path.join(subject, f"{name}_{modality}.nii")
        seg = os.path.join(subject, f"{name}_seg.nii")
        if os.path.exists(img) and os.path.exists(seg):
            images.append(img)
            labels.append(seg)
            patient_ids.append(name)
    n = len(images)
    split = int(n * 0.8)
    return images[split:], labels[split:], patient_ids[split:]

val_images, val_labels, val_patient_ids = build_val_file_lists(DATA_DIR, MODALITY)
print(f"🧪 Validation cases: {len(val_images)}")

# ========================================
# 📋 تحميل البيانات السريرية
# ========================================
df_name = pd.read_csv(NAME_MAP_PATH)
df_surv = pd.read_csv(SURV_PATH)

def lookup_clinical(patient_id: str):
    info = {"Grade": "Unknown", "Age": "Unknown", "Survival_days": "Unknown", "Extent_of_Resection": "Unknown"}
    pid_num = patient_id.replace("BraTS20_Training_", "")

    # name_mapping.csv → Grade
    if "BraTS_2020_subject_ID" in df_name.columns:
        match = df_name[df_name["BraTS_2020_subject_ID"].astype(str).str.contains(pid_num)]
        if not match.empty:
            info["Grade"] = match["Grade"].iloc[0]

    # survival_data.csv → clinical info
    if "Brats20ID" in df_surv.columns:
        match = df_surv[df_surv["Brats20ID"].astype(str).str.zfill(3) == pid_num[-3:]]
        if not match.empty:
            for col in ["Age", "Survival_days", "Extent_of_Resection"]:
                if col in match.columns:
                    info[col] = match[col].iloc[0]

    return info

# ========================================
# 📏 Dice Function
# ========================================
def dice_per_class(pred, gt, num_classes=4, ignore_background=True, eps=1e-6):
    dices = {}
    classes = range(1, num_classes) if ignore_background else range(num_classes)
    for c in classes:
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()
        inter = (pred_c * gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        d = (2.0 * inter + eps) / (denom + eps)
        dices[c] = d.item()
    dices["mean"] = float(np.mean(list(dices.values())))
    return dices

# ========================================
# 🔍 Evaluation Loop
# ========================================
results = []
roi_size = PATCH_SIZE
sw_batch = 2

with torch.no_grad():
    for img_path, seg_path, pid in tqdm(list(zip(val_images, val_labels, val_patient_ids)), desc="Evaluating"):
        # تحميل الصورة والماسك
        img = nib.load(img_path).get_fdata().astype(np.float32)
        gt = nib.load(seg_path).get_fdata().astype(np.float32)
        gt[gt == 4] = 3

        # تطبيع
        img = np.clip(img, 0, 2000) / 2000.0
        img_t = torch.from_numpy(img[None, None]).to(device)

        # التنبؤ
        logits = sliding_window_inference(
            inputs=img_t, roi_size=roi_size, sw_batch_size=sw_batch,
            predictor=model, overlap=0.25
        )
        pred = torch.argmax(logits, dim=1).squeeze(0)
        gt_t = torch.from_numpy(gt).to(device).long()

        # Dice
        dices = dice_per_class(pred.cpu(), gt_t.cpu(), num_classes=4)

        # Visualization (slice وسطية)
        z = img.shape[2] // 2
        flair_slice = img[:, :, z]
        gt_slice = gt[:, :, z]
        pred_slice = pred.cpu().numpy()[:, :, z]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(flair_slice, cmap="gray")
        axes[0].set_title(f"{pid}\nFLAIR (z={z})")
        axes[1].imshow(flair_slice, cmap="gray")
        axes[1].imshow(gt_slice, alpha=0.4)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(flair_slice, cmap="gray")
        axes[2].imshow(pred_slice, alpha=0.4)
        axes[2].set_title(f"Prediction\nMean Dice={dices['mean']:.3f}")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        out_png = os.path.join(VIS_DIR, f"{pid}_z{z:03d}.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)

        # البيانات السريرية
        clin = lookup_clinical(pid)

        results.append({
            "patient_id": pid,
            "dice_core(1)": dices.get(1, 0.0),
            "dice_edema(2)": dices.get(2, 0.0),
            "dice_enh(3)": dices.get(3, 0.0),
            "dice_mean": dices.get("mean", 0.0),
            "Age": clin["Age"],
            "Survival_days": clin["Survival_days"],
            "Extent_of_Resection": clin["Extent_of_Resection"],
            "Grade": clin["Grade"],
            "preview": out_png
        })

# ========================================
# 💾 حفظ النتائج
# ========================================
df = pd.DataFrame(results)
csv_out = os.path.join(MODEL_DIR, "segmentation_eval_with_clinical.csv")
df.to_csv(csv_out, index=False)
print(f"\n✅ Saved evaluation CSV: {csv_out}")
print(f"🖼️ Preview images saved under: {VIS_DIR}")

if len(df) > 0:
    print("\n📈 Summary on validation set:")
    print(df[["dice_core(1)", "dice_edema(2)", "dice_enh(3)", "dice_mean"]].describe().round(4))
    print("\n🏆 Top 5 cases by mean dice:")
    print(df.sort_values('dice_mean', ascending=False).head(5)[["patient_id", "dice_mean", "Grade", "Age", "Survival_days"]])
