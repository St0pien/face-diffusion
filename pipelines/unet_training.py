from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
from model.unet.blocks.time_embedding import encode_timesteps
from data.utils import tensor_to_images
import os
import torch
import torch.nn as nn
import json


class UNetTrainingPipeline:
    def __init__(self,
                 unet: nn.Module,
                 vae: nn.Module,
                 dataloader: DataLoader,
                 device='cuda',
                 ):
        self.device = device
        self.unet = unet.to(device)
        # self.vae = vae.to('device')
        self.dataloader = dataloader

    def train(self,
              name: str,
              output_dir: str,
              epochs: int,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
              n_timesteps=1000,
              sample_period=3,
              ):
        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'samples'))

        loss_per_epoch = []

        noise_scheduler = DDPMScheduler(n_timesteps)
        criterion = nn.MSELoss()

        training_progress = tqdm(range(epochs), position=0)
        try:
            for epoch in training_progress:
                training_progress.set_description("Epoch:")

                self.unet.train()
                running_loss = 0
                epoch_progress = tqdm(self.dataloader, position=1, leave=False)
                for timestep, batch in enumerate(epoch_progress):
                    epoch_progress.set_description("Batch:")
                    batch = batch.to(self.device)
                    noise = torch.randn(batch[0].shape, device=self.device)
                    timesteps = torch.randint(
                        0, n_timesteps, (batch.shape[0],), dtype=torch.int64, device=self.device)
                    noisy_images = noise_scheduler.add_noise(
                        batch, noise, timesteps)
                    enc_timesteps = encode_timesteps(timesteps)

                    noise_pred = self.unet(noisy_images, enc_timesteps)

                    loss = criterion(noise_pred, noise.unsqueeze(
                        0).expand_as(noise_pred))
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
                    self.unet.eval()
                    inputs = torch.randn((16, 3, 32, 32), device=self.device)
                    inference_noise_scheduler = DDPMScheduler(1000)
                    inference_noise_scheduler.set_timesteps(500)
                    sample_progress = tqdm(
                        inference_noise_scheduler.timesteps, position=1, leave=False)
                    sample_progress.set_description("Generating preview: ")
                    for timestep in sample_progress:
                        with torch.no_grad():
                            encoded_timestep = encode_timesteps(
                                torch.Tensor([timestep])).to(self.device)
                            noise_pred = self.unet(inputs, encoded_timestep)
                        inputs = inference_noise_scheduler.step(
                            noise_pred, timestep, inputs).prev_sample
                    output_images = tensor_to_images(inputs)
                    grid = make_image_grid(output_images, rows=4, cols=4)
                    grid.save(f'{output_dir}/samples/{epoch:04d}.png')
        finally:
            torch.save(self.unet.state_dict(),
                       f'{output_dir}/unet.safetensors')
            with open(f'{output_dir}/loss.json', 'w') as f:
                json.dump(loss_per_epoch, f)
            print(f"[+] Training stopped. Results saved to {output_dir}")
