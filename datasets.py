import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils import ImageTransforms


class EnhancedDataset(Dataset):
    def __init__(self, config):
        self.setup_paths(config['data_folder'])
        self.configure_dataset(config['split'], config['crop_size'], config['scaling_factor'])
        self.setup_image_types(config['lr_img_type'], config['hr_img_type'], config.get('test_data_name', None))
        self.load_images()
        self.initialize_transform()

    def setup_paths(self, folder):
        self.data_folder = Path(folder)

    def configure_dataset(self, split, crop_size, scale):
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scale)
        self.validate_configuration()

    def validate_configuration(self):
        valid_splits = {'train', 'test'}
        if self.split not in valid_splits:
            raise ValueError(f"Unsupported dataset split '{self.split}'. Allowed values are {valid_splits}.")

    def setup_image_types(self, lr_type, hr_type, test_name=None):
        valid_types = {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        if lr_type not in valid_types or hr_type not in valid_types:
            raise ValueError(f"Unsupported image type. Choose from {valid_types}.")
        if self.split == 'train' and self.crop_size % self.scaling_factor != 0:
            raise ValueError("Crop size must be divisible by the scaling factor for training.")
        self.lr_img_type = lr_type
        self.hr_img_type = hr_type
        self.test_data_name = test_name

    def load_images(self):
        file_name = 'train_images.json' if self.split == 'train' else f'{self.test_data_name}_test_images.json'
        file_path = self.data_folder / 'TrainJson' / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")
        with open(file_path, 'r') as file:
            self.images = json.load(file)

    def initialize_transform(self):
        self.transform = ImageTransforms(self.split, self.crop_size, self.scaling_factor,
                                         self.lr_img_type, self.hr_img_type)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if min(img.width, img.height) <= 96:
            print(f"Warning: Small image dimensions {img.size} at {img_path}")
        lr_img, hr_img = self.transform(img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.images)
