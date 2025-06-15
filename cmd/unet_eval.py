import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from model.vae.vae import VAE
from model.unet.facediffusion_unet import FaceDiffusionUNet
from model.unet.blocks.time_embedding import encode_timesteps
from data.image_dataset import ImageDataset
from torch.utils.data import Subset
from data.utils import tensor_to_images
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def load_models(vae_path, unet_path):
    vae = VAE()
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()
    unet = FaceDiffusionUNet()
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    return vae, unet


def get_dataloader(image_folder, batch_size=16, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])
    dataset = ImageDataset(image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, prefetch_factor=2)


def evaluate(vae: VAE, unet: FaceDiffusionUNet, dataloader, device):
    is_metric = InceptionScore()
    fid_metric = FrechetInceptionDistance()

    for batch in tqdm(dataloader, desc="Loading real images", leave=True):
        fid_metric.update(batch, real=True)

    context = torch.zeros((16, 512)).to(device)
    convert = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    for i in tqdm(range(0, 1000, 16), desc="Producing fake images", leave=True, position=1):
        with torch.no_grad():
            scheduler = DDPMScheduler(1000)
            scheduler.set_timesteps(30)
            latent = torch.normal(0, 1, (16, 4, 32, 32)).to(device)
            for t in tqdm(scheduler.timesteps, leave=False, position=2):
                encoded_timesteps = encode_timesteps(
                    torch.Tensor([t]).to(device))
                pred_nosise = unet(latent, context, encoded_timesteps)
                latent = scheduler.step(pred_nosise, t, latent).prev_sample
            out_tensor = vae.decoder(latent)
            images = tensor_to_images(out_tensor)
            fid_tensor = torch.stack([convert(i) for i in images])
            fid_metric.update(fid_tensor, real=False)
            is_metric.update(fid_tensor)

    return fid_metric.compute(), is_metric.compute()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction using MSE and MS-SSIM")
    parser.add_argument("--vae-path", type=str, required=True,
                        help="Path to pretrained VAE model")
    parser.add_argument("--unet-path", type=str,
                        required=True, help="Path to pretrained unet model")
    parser.add_argument("--image-folder", type=str,
                        required=True, help="Path to folder with real images")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (resize to this before feeding to model)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    vae, unet = load_models(args.vae_path, args.unet_path)
    vae.to(device)
    unet.to(device)

    print("Loading data...")
    dataloader = get_dataloader(
        args.image_folder, batch_size=args.batch_size, image_size=args.image_size)

    print("Evaluating...")
    fid, (is_mean, is_std) = evaluate(vae, unet, dataloader, device)

    print(f"\nFID score: {fid:.6f}")
    print(f"IS score: {is_mean:.6f} +- {is_std:.6f}")


if __name__ == "__main__":
    main()
