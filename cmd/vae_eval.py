import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
from model.vae.vae import VAE
from data.image_dataset import ImageDataset


def load_model(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_dataloader(image_folder, batch_size=16, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(model: VAE, dataloader, device):
    mse_metric = MeanSquaredError().to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)

            # Pass through VAE
            seed = torch.normal(0, 1, size=(
                batch.shape[0], 4, 32, 32)).to(device)
            recon, _, _ = model(batch, seed)

            # MSE
            mse_metric.update(recon, batch)
            # MS-SSIM
            ms_ssim_metric.update(recon, batch)

    return mse_metric.compute(), ms_ssim_metric.compute()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction using MSE and MS-SSIM")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to pretrained VAE model")
    parser.add_argument("--image-folder", type=str,
                        required=True, help="Path to folder with real images")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (resize to this before feeding to model)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(args.model_path)
    model.to(device)

    print("Loading data...")
    dataloader = get_dataloader(
        args.image_folder, batch_size=args.batch_size, image_size=args.image_size)

    print("Evaluating...")
    mse, ms_ssim = evaluate(model, dataloader, device)

    print(f"\nAverage MSE: {mse:.6f}")
    print(f"Average MS-SSIM: {ms_ssim:.6f}")


if __name__ == "__main__":
    main()
