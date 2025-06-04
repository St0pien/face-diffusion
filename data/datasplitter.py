import kagglehub
from torchvision.datasets import ImageFolder
from pathlib import Path
import torch
from torch.utils.data import random_split
from shutil import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm


default_splits = {
    'VAE_train': 0.4 * 0.7,
    'VAE_val': 0.4 * 0.2,
    'VAE_test': 0.4 * 0.1,
    'UNet_train': 0.6 * 0.7,
    'UNet_val': 0.6 * 0.2,
    'UNet_test': 0.6 * 0.1,
}


class DatasetSplitter:
    def __init__(self, splits: dict = default_splits, seed=None):
        self.splits = splits
        self.generator = torch.Generator().manual_seed(
            seed) if seed is not None else None

    def download_from_kaggle(self):
        return kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")

    def split_dataset(self, output_dir, max_workers=16):
        input_dir = Path(self.download_from_kaggle())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [path for path, _ in ImageFolder(input_dir).imgs]
        print(f"[+] {len(all_imgs)} images loaded")

        _datasets = random_split(
            all_imgs, self.splits.values(), generator=self.generator)
        datasets = dict((name, _datasets[i])
                        for i, name in enumerate(self.splits.keys()))

        for subdir in self.splits.keys():
            Path(output_dir / subdir).mkdir()

        jobs = [(Path(img), output_dir / name / Path(img).name)
                for name, imgs in datasets.items() for img in imgs]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(copy, src, dst) for src, dst in jobs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying images"):
                pass

        print("[+] Split successful:")
        for name, dataset in datasets.items():
            print(
                f"\t{name}: {len(dataset)} images in {Path(output_dir / name).resolve()}")
