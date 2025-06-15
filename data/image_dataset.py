from torch.utils.data import Dataset
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.all_imgs = [os.path.join(root, f) for f in os.listdir(root)]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        path = self.all_imgs[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image


class ImageCLIPDataset(Dataset):
    def __init__(self, root, transform=None, clip_preprocess=None):
        super().__init__()
        self.root = root
        self.transform = transform
        if clip_preprocess is None:
            raise Exception("clip_preprocess not set")
        self.clip_preprocess = clip_preprocess
        self.all_imgs = [os.path.join(root, f) for f in os.listdir(root)]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        path = self.all_imgs[index]
        image = Image.open(path).convert('RGB')
        clip_data = self.clip_preprocess(image)
        if self.transform:
            image = self.transform(image)

        return image, clip_data
