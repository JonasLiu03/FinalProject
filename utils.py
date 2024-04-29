from PIL import Image
import random
import torchvision.transforms.functional as FT
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
RGB_WEIGHTS = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def convert_image(img, source, target):
    assert source in {'pil', '[0, 1]', '[-1, 1]'}
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'}

    conversion_map = {
        'pil': lambda x: FT.to_tensor(x),
        '[0, 1]': lambda x: x,
        '[-1, 1]': lambda x: (x + 1) / 2
    }

    img = conversion_map[source](img)

    transformation_map = {
        'pil': lambda x: FT.to_pil_image(x.clamp(0, 1)),
        '[0, 255]': lambda x: 255 * x,
        '[0, 1]': lambda x: x,
        '[-1, 1]': lambda x: 2 * x - 1,
        'imagenet-norm': lambda x: (x - IMAGENET_MEAN) / IMAGENET_STD,
        'y-channel': lambda x: torch.matmul(255 * x.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], RGB_WEIGHTS) / 255 + 16
    }

    return transformation_map[target](img)


class ImageTransforms(object):
    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        if self.split == 'train':
            # Random crop for training images
            left = random.randint(0, img.width - self.crop_size)
            top = random.randint(0, img.height - self.crop_size)
            hr_img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        else:
            # Center crop for testing images
            left = (img.width % self.scaling_factor) // 2
            top = (img.height % self.scaling_factor) // 2
            hr_img = img.crop((left, top, img.width - left, img.height - top))

        # Downscale and upscale to simulate low-resolution creation
        lr_img = hr_img.resize((hr_img.width // self.scaling_factor, hr_img.height // self.scaling_factor),
                               Image.BICUBIC)
        hr_img = hr_img.resize((lr_img.width * self.scaling_factor, lr_img.height * self.scaling_factor), Image.BICUBIC)

        # Convert images to desired formats
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
