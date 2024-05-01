from PIL import Image
import random
import torchvision.transforms.functional as FT
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
RGB_WEIGHTS = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
IMAGENET_STATS = {
    'mean': torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3),
    'std': torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
}

def convert_image(img, source, target):
    assert source in {'pil', '[0, 1]', '[-1, 1]'}
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'}

    conversions = {
        'pil': lambda x: FT.to_tensor(x),
        '[0, 1]': lambda x: x,
        '[-1, 1]': lambda x: (x + 1) / 2
    }

    transformations = {
        'pil': lambda x: FT.to_pil_image(x.clamp(0, 1)),
        '[0, 255]': lambda x: 255 * x,
        '[0, 1]': lambda x: x,
        '[-1, 1]': lambda x: 2 * x - 1,
        'imagenet-norm': lambda x: (x - IMAGENET_STATS['mean']) / IMAGENET_STATS['std'],
        'y-channel': lambda x: torch.matmul(255 * x.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], RGB_WEIGHTS) / 255 + 16
    }

    img = conversions[source](img)
    return transformations[target](img)

class ImageTransforms:
    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

    def __call__(self, img):
        crop_fn = random_crop if self.split == 'train' else center_crop
        hr_img = crop_fn(img, self.crop_size, self.scaling_factor)
        lr_img = resize_image(hr_img, self.scaling_factor, method=Image.BICUBIC)
        hr_img = resize_image(lr_img, self.scaling_factor, scale_up=True, method=Image.BICUBIC)
        return convert_image(lr_img, 'pil', self.lr_img_type), convert_image(hr_img, 'pil', self.hr_img_type)

def random_crop(img, crop_size, scaling_factor):
    left = random.randint(0, img.width - crop_size)
    top = random.randint(0, img.height - crop_size)
    return img.crop((left, top, left + crop_size, top + crop_size))

def center_crop(img, crop_size, scaling_factor):
    left = (img.width % scaling_factor) // 2
    top = (img.height % scaling_factor) // 2
    right = left + (img.width - left * 2)
    bottom = top + (img.height - top * 2)
    return img.crop((left, top, right, bottom))

def resize_image(img, scaling_factor, scale_up=False, method=Image.BICUBIC):
    if scale_up:
        return img.resize((img.width * scaling_factor, img.height * scaling_factor), method)
    return img.resize((img.width // scaling_factor, img.height // scaling_factor), method)

class AverageMeter:
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
        param_group['lr'] *= shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
