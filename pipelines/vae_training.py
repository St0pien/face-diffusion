from torch.utils.data import DataLoader
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
from data.utils import tensor_to_images
import os
import torch
import torch.nn as nn
import json


class VAETrainingPipeline:
    def __init__(self,
                 vae: nn.Module,
                 dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 device='cuda',
                 ):
        self.device = device
        self.vae = vae.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

    def train(self,
              name: str,
              output_dir: str,
              epochs: int,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
              sample_period=3,
              ):
        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'samples'))

        loss_per_epoch = []

        mse = nn.MSELoss()
        val_batch = next(iter(self.val_dataloader)).to(self.device)

        training_progress = tqdm(range(epochs), position=0)
        try:
            for epoch in training_progress:
                training_progress.set_description("Epoch:")

                self.vae.train()
                running_loss = 0
                epoch_progress = tqdm(self.dataloader, position=1, leave=False)
                for timestep, batch in enumerate(epoch_progress):
                    epoch_progress.set_description("Batch:")
                    batch = batch.to(self.device)

                    seed = torch.normal(0, 1, size=(
                        batch.shape[0], 4, batch.shape[2]//8, batch.shape[3]//8), device=self.device)

                    decoded_image, mean, logvar = self.vae(batch, seed)

                    rec_loss = mse(decoded_image, batch)
                    kl_loss = -0.5 * \
                        torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                    loss = rec_loss + kl_loss * 1e-8
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    postfix = {'loss': loss.item()}
                    if lr_scheduler is not None:
                        postfix['lr'] = lr_scheduler.get_last_lr()[0]
                    epoch_progress.set_postfix(postfix)

                loss_per_epoch.append(running_loss / len(self.dataloader))
                if (epoch + 1) % sample_period == 0:
                    self.vae.eval()
                    images = val_batch
                    seed = torch.normal(0, 1, size=(
                        8, 4, images.shape[2]//8, images.shape[3]//8)).to(self.device)
                    with torch.no_grad():
                        decoded_images = self.vae(images, seed)[0]

                    output_images = tensor_to_images(torch.stack(
                        [images, decoded_images], dim=1).view(16, 3, images.shape[2], images.shape[3]))
                    grid = make_image_grid(output_images, rows=4, cols=4)
                    grid.save(f'{output_dir}/samples/{epoch:04d}.png')

        finally:
            torch.save(self.vae.state_dict(),
                       f'{output_dir}/vae.safetensors')
            with open(f'{output_dir}/loss.json', 'w') as f:
                json.dump(loss_per_epoch, f)
            print(f"[+] Training stopped. Results saved to {output_dir}")
