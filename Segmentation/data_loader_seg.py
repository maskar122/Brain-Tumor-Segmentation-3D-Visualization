# ===============================
# 📦 Data Loader for BraTS2020 Segmentation (Final Clean Version)
# ===============================

import os
import glob
import torch
import nibabel as nib
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from config_seg import DATA_DIR, PATCH_SIZE, BATCH_SIZE, MODALITY


# ===============================
# 🧩 Custom Dataset Class
# ===============================
class BraTSDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # تحميل الصورة والماسك
        image = nib.load(image_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)

        # 🧩 إصلاح قيم الماسك: تحويل 4 → 3
        label[label == 4] = 3

        # تطبيع الصورة
        image = np.clip(image, 0, 2000)
        image = image / 2000.0

        # إضافة بعد القناة
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # تطبيق التحويلات (لو موجودة)
        if self.transforms:
            image, label = self.transforms(image, label)

        return torch.tensor(image), torch.tensor(label)


# ===============================
# ⚙️ Function to create perfectly matched datasets
# ===============================
def get_datasets():
    # جمع كل المجلدات (كل مريض)
    subject_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS20_Training_*")))

    images = []
    labels = []

    for subject in subject_dirs:
        subject_name = os.path.basename(subject)
        flair_path = os.path.join(subject, f"{subject_name}_{MODALITY}.nii")
        seg_path = os.path.join(subject, f"{subject_name}_seg.nii")

        # نضيف بس الحالات اللي فيها الصورتين
        if os.path.exists(flair_path) and os.path.exists(seg_path):
            images.append(flair_path)
            labels.append(seg_path)

    n = len(images)
    split = int(n * 0.8)
    train_images, val_images = images[:split], images[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print(f"✅ Found {len(train_images)} training cases and {len(val_images)} validation cases (Perfectly matched).")

    # 🧠 Transform بسيط (قص عشوائي)
    def simple_transform(image, label):
        x, y, z = image.shape[1:]
        start_x = np.random.randint(0, x - PATCH_SIZE[0])
        start_y = np.random.randint(0, y - PATCH_SIZE[1])
        start_z = np.random.randint(0, z - PATCH_SIZE[2])

        image_crop = image[:, start_x:start_x + PATCH_SIZE[0],
                           start_y:start_y + PATCH_SIZE[1],
                           start_z:start_z + PATCH_SIZE[2]]
        label_crop = label[:, start_x:start_x + PATCH_SIZE[0],
                           start_y:start_y + PATCH_SIZE[1],
                           start_z:start_z + PATCH_SIZE[2]]

        return image_crop, label_crop

    # إنشاء Datasets
    train_ds = BraTSDataset(train_images, train_labels, transforms=simple_transform)
    val_ds = BraTSDataset(val_images, val_labels, transforms=None)

    # إنشاء DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader


# ===============================
# 🔍 اختبار سريع (للتأكد من أن الداتا سليمة)
# ===============================
if __name__ == "__main__":
    train_loader, val_loader = get_datasets()

    print("\n🎯 Checking a few random samples from training data:\n")
    all_unique = set()

    for i in range(5):
        idx = random.randint(0, len(train_loader.dataset) - 1)
        img, mask = train_loader.dataset[idx]
        uniques = torch.unique(mask)
        all_unique.update(uniques.tolist())
        print(f"🧩 Sample {i+1} unique labels: {uniques.tolist()}")

    print("\n🎯 All labels found across 5 samples:", sorted(list(all_unique)))
    print("✅ If you see [0.0, 1.0, 2.0, 3.0], your masks are perfect!\n")
