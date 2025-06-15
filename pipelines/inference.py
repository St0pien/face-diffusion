from model.vae.vae import VAE
from model.unet.facediffusion_unet import FaceDiffusionUNet
from diffusers.schedulers import DPMSolverSinglestepScheduler, DDPMScheduler
from model.unet.blocks.time_embedding import encode_timesteps
from tqdm.auto import tqdm
from data.utils import tensor_to_images
import open_clip
import torch


class FaceDiffusionInferencePipeline:
    def __init__(self, vae: VAE, unet: FaceDiffusionUNet, device='cuda'):
        self.device = device
        self.vae = vae.to(device)
        self.unet = unet.to(device)
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def generate(
            self,
            prompt: str,
            timesteps: int,
            batch_size=4,
            cfg_scale=7.5
    ):
        with torch.no_grad():
            self.vae.eval()
            self.unet.eval()
            tokens = self.tokenizer(prompt).to(self.device)
            conditioning = self.clip.encode_text(tokens)
            conditioning = conditioning.expand(
                (batch_size, conditioning.shape[1]))
            context = torch.cat((conditioning, torch.zeros_like(conditioning)))
            scheduler = DDPMScheduler(1000)
            # scheduler = DPMSolverSinglestepScheduler(
            #     1000, solver_order=2, lower_order_final=True, use_karras_sigmas=False)
            scheduler.set_timesteps(timesteps)
            latent = torch.normal(
                0, 1, (batch_size, 4, 32, 32)).to(self.device)

            for t in tqdm(scheduler.timesteps):
                encoded_timestep = encode_timesteps(
                    torch.Tensor([t]).to(self.device))
                model_input = latent.repeat(2, 1, 1, 1)
                pred_noise = self.unet(model_input, context, encoded_timestep)
                noise_cond, noise_uncond = pred_noise.chunk(2)
                pred_noise = cfg_scale * \
                    (noise_cond - noise_uncond) + noise_uncond
                latent = scheduler.step(pred_noise, t, latent).prev_sample

            img_tensor = self.vae.decoder(latent)
            return tensor_to_images(img_tensor)
