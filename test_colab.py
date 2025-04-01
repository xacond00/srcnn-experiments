import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
import numpy as np
import torch
from ssim import ssim
from IPython.display import display
from PIL import Image

print(torch.cuda.is_available())


def compute_psnr(gt, pred):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = ((gt - pred) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Pokud jsou obrázky normalizované (0,1), jinak použij 255.0
    return 10 * np.log10(max_pixel**2 / mse)


def compute_ssim(gt, pred):
    """Compute Structural Similarity Index (SSIM)."""
    return ssim(
        gt, pred,
        window_size=11,
        in_channels=gt.shape[1],
        L=1  # Change to 255 if your images are in the 0-255 range
    ).item()


# Load models and their respective checkpoints
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = checkpoint['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader):
    model.eval()
    psnr_values = []
    ssim_values = []
    images = None
    for n, (lr, hr) in enumerate(dataloader):
        lr = lr.to(next(model.parameters()).device)  # Přenos na správné zařízení
        hr = hr.to(next(model.parameters()).device)

        with torch.no_grad():
            sr = model(lr)
        sr = torch.clip(sr, 0, 1)
        #sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if(n < 1):
            images = sr
        psnr_values.append(compute_psnr(hr, sr))
        ssim_values.append(compute_ssim(hr, sr))
    return images, np.mean(psnr_values), np.mean(ssim_values)

if __name__ == "__main__":
        
    model_checkpoints = [
        "resnet_correct.pth",
        #"./final/4x64mae_c5x2_c3x5.pth",
        #"./final/4x64ssae_c5x2_c3x5.pth",
        #"./final/4x64ssim_c5x2_c3x5.pth",
        #"./final/4x64uvge_c5x2_c3x5.pth",
        #"./ssresnet_ssae.pth"
    ]
    scale = 4
    batch = 16

    # Load dataset
    dataset = ImageDataset(dataset_name="DIV2KVal", train=False, scale=scale, downscale=1, crop=512)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)

    # Run evaluation
    results = {}
    sr_imgs = []
    for checkpoint in model_checkpoints:
        model = load_model(checkpoint)
        images, mean_psnr, mean_ssim = evaluate_model(model, dataloader)
        sr_imgs.append(images)
        results[checkpoint] = {"PSNR": mean_psnr, "SSIM": mean_ssim}
        print(f"{checkpoint}: PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}")
        
    scaler = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=True)
    for n in range(batch):
        lr, hr = dataset[n]
        lr = scaler(lr.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        hr = hr.permute(1, 2, 0).cpu().numpy()
        all_imgs = [lr]
        for sr_img in sr_imgs:
            all_imgs.append(sr_img[n].permute(1, 2, 0).cpu().numpy())
        all_imgs.append(hr)
        res = np.hstack(all_imgs)
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        display(Image.fromarray(res, 'RGB'))

    # Store results
    with open("evaluation_results.txt", "w") as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}\n")

    print("Evaluation complete. Results saved to evaluation_results.txt")
