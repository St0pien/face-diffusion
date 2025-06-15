from data.image_dataset import ImageCLIPDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model.vae.vae import VAE
from model.unet.facediffusion_unet import FaceDiffusionUNet
from torch.optim import AdamW
from diffusers import get_cosine_schedule_with_warmup
from torch.nn import MSELoss
from pipelines.unet_training import UNetTrainingPipeline
import torchvision.transforms as transforms
import open_clip
import torch

parser = ArgumentParser(prog="Train unet with VAE", description="")
parser.add_argument("--dataset", help='training dataset',
                    default='./datasets2/unet_train')
parser.add_argument("--vae", help="path to pretrained VAE", required=True)
parser.add_argument("--unet", help="path to pretrained UNet", required=True)
parser.add_argument("--epochs", required=True)
parser.add_argument("--name", help="name of the rung", required=True)

options = parser.parse_args()
epochs = int(options.epochs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
clip.eval()

dataset = ImageCLIPDataset(options.dataset, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]), clip_preprocess=preprocess)

dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=16, prefetch_factor=2)


vae = VAE().to(device)
vae.load_state_dict(torch.load(options.vae))
vae.eval()

unet = FaceDiffusionUNet().to(device)
if options.unet is not None:
    unet.load_state_dict(torch.load(options.unet))
    print(f"[+] Pretrained unet loaded from {options.unet}")


optimizer = AdamW(unet.parameters(), lr=1e-5)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=epochs * len(dataloader))
criterion = MSELoss()

pipeline = UNetTrainingPipeline(
    unet=unet,
    vae=vae,
    dataloader=dataloader,
    clip=clip,
    device=device
)

pipeline.train(options.name, './runs', epochs, optimizer=optimizer,
               lr_scheduler=lr_scheduler, sample_period=1)
