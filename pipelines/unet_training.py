from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
from model.unet.blocks.time_embedding import encode_timesteps
from model.vae.vae import VAE
from data.utils import tensor_to_images
from open_clip import CLIP
import math
import os
import torch
import torch.nn as nn
import json


class UNetTrainingPipeline:
    def __init__(self,
                 unet: nn.Module,
                 vae: VAE,
                 clip: CLIP,
                 dataloader: DataLoader,
                 device='cuda',
                 ):
        self.device = device
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.dataloader = dataloader
        self.clip = clip.to(device)

    def train(self,
              name: str,
              output_dir: str,
              epochs: int,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
              n_timesteps=1000,
              sample_period=3,
              fraction_with_zero_context=0.3
              ):
        self.vae.eval()
        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'samples'))

        loss_per_epoch = []
        loss_per_step = []

        noise_scheduler = DDPMScheduler(n_timesteps)
        criterion = nn.MSELoss()

        training_progress = tqdm(range(epochs), position=0)
        try:
            for epoch in training_progress:
                training_progress.set_description("Epoch:")

                self.unet.train()
                running_loss = 0
                epoch_progress = tqdm(self.dataloader, position=1, leave=False)
                for timestep, (batch, batch_clip) in enumerate(epoch_progress):
                    epoch_progress.set_description("Batch:")
                    batch = batch.to(self.device)
                    encode_seed = torch.normal(0, 1, size=(
                        1, 4, batch.shape[2] // 8, batch.shape[3]//8)).to(self.device)
                    with torch.no_grad():
                        latent_input = self.vae.encoder(batch, encode_seed)[0]

                    noise = torch.randn(
                        latent_input[0].shape, device=self.device)
                    timesteps = torch.randint(
                        0, n_timesteps, (latent_input.shape[0],), dtype=torch.int64, device=self.device)
                    noisy_latents = noise_scheduler.add_noise(
                        latent_input, noise, timesteps)
                    enc_timesteps = encode_timesteps(timesteps)

                    clip_context = self.clip.encode_image(
                        batch_clip.to(self.device))
                    no_context_size = math.floor(
                        clip_context.shape[0] * fraction_with_zero_context)
                    no_context = torch.randperm(clip_context.shape[0])[
                        :no_context_size]
                    clip_context[no_context] = torch.zeros_like(
                        clip_context[no_context])

                    noise_pred = self.unet(
                        noisy_latents, clip_context, enc_timesteps)

                    loss = criterion(noise_pred, noise.unsqueeze(
                        0).expand_as(noise_pred))
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    loss_per_step.append(loss.item())
                    postfix = {'loss': loss.item()}
                    if lr_scheduler is not None:
                        postfix['lr'] = lr_scheduler.get_last_lr()[0]
                    epoch_progress.set_postfix(postfix)

                loss_per_epoch.append(running_loss / len(self.dataloader))
                if (epoch + 1) % sample_period == 0:
                    self.unet.eval()
                    inputs = torch.randn((16, 4, 32, 32), device=self.device)
                    inference_noise_scheduler = DDPMScheduler(1000)
                    inference_noise_scheduler.set_timesteps(500)
                    sample_progress = tqdm(
                        inference_noise_scheduler.timesteps, position=1, leave=False)
                    sample_progress.set_description("Generating preview: ")
                    for timestep in sample_progress:
                        with torch.no_grad():
                            encoded_timestep = encode_timesteps(
                                torch.Tensor([timestep])).to(self.device)
                            prompt = torch.zeros((16, 512), device=self.device)
                            noise_pred = self.unet(
                                inputs, prompt, encoded_timestep)
                        inputs = inference_noise_scheduler.step(
                            noise_pred, timestep, inputs).prev_sample

                    with torch.no_grad():
                        output_tensors = self.vae.decoder(inputs)

                    output_images = tensor_to_images(output_tensors)
                    grid = make_image_grid(output_images, rows=4, cols=4)
                    grid.save(f'{output_dir}/samples/{epoch:04d}.png')
        finally:
            torch.save(self.unet.state_dict(),
                       f'{output_dir}/unet.safetensors')
            with open(f'{output_dir}/loss_epoch.json', 'w') as f:
                json.dump(loss_per_epoch, f)
            with open(f'{output_dir}/loss_step.json', 'w') as f:
                json.dump(loss_per_step, f)
            print(f"[+] Training stopped. Results saved to {output_dir}")
