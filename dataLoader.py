from gettext import npgettext
import os
from PIL import Image
from torch.utils.data import Dataset 
import numpy as np


class CaravanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # lists all the files in image directiry

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])#.replace(".jpg", "_mask.gif")) # replacing the .gif with .jpg for mask img

        image = np.array(Image.open(img_path).convert("RGB")) # We use np array because if using albumentaion linrary we need to convert to array becuase of PIL

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Now we create some sort of preprocess for the mask, we look for where the pixel val =255 and change to 1
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

print("Testing!")