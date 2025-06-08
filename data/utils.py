from PIL import Image
import torch


def tensor_to_images(tensor: torch.Tensor):
    reshaped = tensor.cpu().permute(0, 2, 3, 1)
    denormalized = ((reshaped + 1.0)*127.5).type(torch.uint8)
    return [Image.fromarray(t.numpy()) for t in denormalized]
