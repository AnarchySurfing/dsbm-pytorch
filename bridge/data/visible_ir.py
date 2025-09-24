import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class VisibleIR(Dataset):
    def __init__(self, root_dir, transform=None, domain='visible', return_path=False):
        self.root_dir = root_dir
        self.transform = transform
        self.domain = domain
        self.return_path = return_path

        if self.domain not in ['visible', 'infrared']:
            raise ValueError("Domain must be either 'visible' or 'infrared'.")

        self.visible_dir = os.path.join(self.root_dir, 'visible')
        self.infrared_dir = os.path.join(self.root_dir, 'infrared')

        self.visible_images = sorted(os.listdir(self.visible_dir))
        self.infrared_images = sorted(os.listdir(self.infrared_dir))

        # Sanity check to ensure the datasets are aligned
        if len(self.visible_images) != len(self.infrared_images) or self.visible_images != self.infrared_images:
            raise ValueError("Visible and Infrared directories must contain the same number of images with matching filenames.")

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        if self.domain == 'visible':
            img_path = os.path.join(self.visible_dir, self.visible_images[idx])
        else: # self.domain == 'infrared'
            img_path = os.path.join(self.infrared_dir, self.infrared_images[idx])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, img_path
        else:
            return image
