import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils import ImageTransforms


class SRDataset(Dataset):
    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        self.data_folder = Path(data_folder)
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        self.validate_inputs()
        self.images = self.load_image_paths()
        self.transform = self.select_transforms()

    def validate_inputs(self):
        valid_splits = {'train', 'test'}
        valid_img_types = {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        if self.split not in valid_splits:
            raise ValueError(f"Invalid split {self.split}. Choose from {valid_splits}.")
        if self.lr_img_type not in valid_img_types or self.hr_img_type not in valid_img_types:
            raise ValueError(f"Invalid image type. Choose from {valid_img_types}.")
        if self.split == 'train' and self.crop_size % self.scaling_factor != 0:
            raise ValueError("Crop size must be divisible by scaling factor in training mode.")

    def load_image_paths(self):
        file_path = self.data_folder / (
            'TrainJson/train_images.json' if self.split == 'train' else f'{self.test_data_name}_test_images.json')
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")
        with open(file_path, 'r') as file:
            return json.load(file)

    def select_transforms(self):
        return ImageTransforms(split=self.split, crop_size=self.crop_size, scaling_factor=self.scaling_factor,
                               lr_img_type=self.lr_img_type, hr_img_type=self.hr_img_type)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(f"Image {img_path} is too small with dimensions ({img.width}, {img.height}).")
        lr_img, hr_img = self.transform(img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.images)
