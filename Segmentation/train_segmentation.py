# ===============================
# 🧱 Training Script for BraTS2020 3D Segmentation
# ===============================

import os
import torch
from tqdm import tqdm
import numpy as np
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from config_seg import EPOCHS, MODEL_DIR, MODEL_NAME
from data_loader_seg import get_datasets
from model_seg import get_unet3d, get_loss_optimizer


# -------------------------------
# ⚙️ إعداد الجهاز (GPU أو CPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# -------------------------------
# 📦 تحميل البيانات والموديل
# -------------------------------
train_loader, val_loader = get_datasets()
model = get_unet3d()
loss_function, optimizer = get_loss_optimizer(model)
model = model.to(device)

dice_metric = DiceMetric(include_background=False, reduction="mean")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 🧠 وظيفة مساعدة لحفظ النموذج الأفضل
# -------------------------------
def save_model(model, epoch, dice_score):
    model_path = os.path.join(MODEL_DIR, f"unet3d_epoch{epoch+1}_dice{dice_score:.3f}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")


# -------------------------------
# 🎯 التدريب
# -------------------------------
best_dice = 0.0
for epoch in range(EPOCHS):
    print(f"\n========== Epoch [{epoch+1}/{EPOCHS}] ==========")
    model.train()
    epoch_loss = 0

    for i, batch_data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"📉 Average Training Loss: {avg_loss:.4f}")

    # -------------------------------
    # 🧪 التقييم على الـValidation
    # -------------------------------
    model.eval()
    val_dice = []

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="Validating", leave=False):
            val_inputs, val_labels = val_data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # inference window لتقليل استهلاك الذاكرة
            val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model)
            val_outputs = [torch.argmax(o, dim=1, keepdim=True) for o in decollate_batch(val_outputs)]

            dice_metric(y_pred=val_outputs, y=val_labels)
        
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()

    print(f"🎯 Validation Mean Dice: {mean_dice:.4f}")

    # حفظ النموذج الأفضل
    if mean_dice > best_dice:
        best_dice = mean_dice
        save_model(model, epoch, mean_dice)

print(f"\n✅ Training Completed. Best Dice Score: {best_dice:.4f}")
