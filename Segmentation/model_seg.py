# ===============================
# 🧠 Model Builder for BraTS2020 Segmentation
# ===============================
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from config_seg import LEARNING_RATE


# ===============================
# 🧱 Build UNet-3D Model
# ===============================
def get_unet3d(in_channels=1, out_channels=4):
    """
    Create a 3D U-Net model for brain tumor segmentation.
    """
    model = UNet(
        spatial_dims=3,       # 3D model
        in_channels=in_channels,  # Input channels (1 for FLAIR)
        out_channels=out_channels,  # Output channels (4 tumor regions)
        channels=(16, 32, 64, 128, 256),  # feature map sizes
        strides=(2, 2, 2, 2),  # downsampling strides
        num_res_units=2,        # residual blocks per level
        norm="batch",           # normalization type
    )

    return model


# ===============================
# ⚙️ Loss & Optimizer
# ===============================
def get_loss_optimizer(model):
    """
    Define Dice loss and optimizer.
    """
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return loss_function, optimizer


# ===============================
# 🔍 اختبار سريع
# ===============================
if __name__ == "__main__":
    model = get_unet3d()
    loss_function, optimizer = get_loss_optimizer(model)

    x = torch.randn(1, 1, 128, 128, 128)  # Batch=1, 1 channel, 3D image
    out = model(x)
    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", out.shape)
