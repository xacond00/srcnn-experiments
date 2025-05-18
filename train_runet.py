# Trained model available at: https://drive.google.com/file/d/1tpOeYo_IzkXWSI9bAcSNOqRaBURMTLuF/view?usp=drive_link

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import ssim
import os
import random
import numpy as np
from runet import RUNet
from dataset import ImageDataset
from train import train


torch.cuda.empty_cache()
# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 300
batch_size = 4
lr = 1.0e-4
grad_clip = 1.0
print_freq = 10
scale_factor = 4
valid_size = 8 # Validation batch
valid_crop = 128 # Validation crop
pre_scale = 1 # Prescale in training
ssim_alpha = 0.5  # Mix mae with vgg

crop_size = 128
dataset_name = "DIV2K"
model_save_path = "test.pth"
seed = 42

samples_dir = "samples"
os.makedirs(samples_dir, exist_ok=True)

# Save image showcasing training progress
def save_sample(model, dataset, device, epoch, scale_factor, val_loss, model_save_path):
    model.eval()

    idx = random.randint(0, len(dataset) - 1)

    lr, hr = dataset.load_img(idx, scale_factor, 1, 256, False)
    sr_in = lr.unsqueeze(0).to(device, memory_format=torch.channels_last)
    
    with torch.no_grad():
        sr = model(sr_in).squeeze()
        sr = torch.clip(sr, 0, 1)

    sr_np = sr.permute(1, 2, 0).cpu().numpy()
    lr_np = lr.permute(1, 2, 0).cpu().numpy()
    hr_np = hr.permute(1, 2, 0).cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(lr_np)
    axes[0].axis("off")
    axes[0].set_title("Low-Resolution (LR)")

    axes[1].imshow(sr_np)
    axes[1].axis("off")
    axes[1].set_title("Super-Resolution (SR)")

    axes[2].imshow(hr_np)
    axes[2].axis("off")
    axes[2].set_title("High-Resolution (HR)")

    plt.tight_layout()

    filename = os.path.join(samples_dir, f"model-{model_save_path}-epoch{epoch}-loss-{val_loss:.4f}.png")
    plt.savefig(filename)
    plt.close(fig)

    # print(f"Saved sample image to {filename}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# VGG19 for perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        loss = nn.functional.mse_loss(x_features, y_features)
        return loss

def main():
    print(torch.cuda.is_available())
    # seed_everything(seed)

    # === DATASETS ===
    train_dataset = ImageDataset(
        dataset_name=dataset_name, 
        train=True, 
        scale=scale_factor, 
        crop=crop_size, 
        cache_size='quar'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RUNet(upscale_factor=scale_factor).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_x = []
    valid_y = []
    for idx in range(valid_size):
        x, y = train_dataset.load_img(idx, scale_factor, pre_scale, valid_crop, False)
        print(x.size(), y.size())
        valid_x.append(x)
        valid_y.append(y)
    if valid_size:
        valid_x = torch.stack(valid_x).to(device, memory_format=torch.channels_last)
        valid_y = torch.stack(valid_y).to(device, memory_format=torch.channels_last)
        valid_ds = (valid_x, valid_y)
    else:
        valid_ds = None

    # === LOSS, OPTIMIZER ===
    # l1_loss_fn = nn.L1Loss(reduction='mean')
    # perceptual_loss = PerceptualLoss().to(device)

    # def combined_loss(output, target):
    #     # print("Shapes ......", output.shape, target.shape)
    #     return l1_loss_fn(output, target) + 0.003 * perceptual_loss(output, target)

    # criterion = combined_loss
    # criterion = l1_loss_fn

    criterion = ssim.SSIM(in_channels=3, as_loss=True, mae_alpha=ssim_alpha)
    criterion.to(device, memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # === LEARNING RATE SCHEDULER ===
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # === TRAINING LOOP ===
    best_loss = float('inf')

    print("Beggining trainig")
    for epoch in range(1, epochs + 1):
        val_loss = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            grad_clip=grad_clip,
            print_freq=print_freq,
            device=device,
            valid_ds=valid_ds
        )

        # Save best model
        if val_loss < best_loss:
            print(f"Saving new best model with val loss: {val_loss:.4f}")
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            # Save image showcasing training progress
            save_sample(model, train_dataset, device, epoch, scale_factor, val_loss, model_save_path)

        scheduler.step()

    print("Training complete.")
    torch.save(model.state_dict(), f"{model_save_path}-complete")
    # save_sample(model, train_dataset, device, epoch, scale_factor, val_loss, model_save_path)

if __name__ == '__main__':
    main()