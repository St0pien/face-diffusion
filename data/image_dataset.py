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
