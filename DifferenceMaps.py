import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont, ImageChops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet_checkpoint = "SrresnetCheckpoint/checkpoint_epoch_42_srresnet.pth.tar"
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()

def visualize_sr_modified(img, halve=False):
    hr_img = Image.open(img).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)

    # 假设 srresnet 返回的是一个元组，其中第一个元素是图像
    output_tuple = srresnet(convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = output_tuple[0].squeeze(0).cpu().detach()  # 只对第一个输出应用处理
    sr_img_srresnet = convert_image(sr_img_srresnet, '[-1, 1]', 'pil')

    # 计算与原图的差异映射
    diff_sr = ImageChops.difference(hr_img, sr_img_srresnet).convert('L')
    diff_bicubic = ImageChops.difference(hr_img, bicubic_img).convert('L')

    margin = 40
    grid_img = Image.new('RGB', (5 * hr_img.width + 6 * margin, hr_img.height + 2 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("calibril.ttf", 23)
    except OSError:
        font = ImageFont.load_default()


    grid_img.paste(hr_img, (margin, margin))
    draw.text((margin, margin - 30), "Original HR", fill="black", font=font)

    grid_img.paste(bicubic_img, (2 * margin + hr_img.width, margin))
    draw.text((2 * margin + hr_img.width, margin - 30), "Bicubic", fill="black", font=font)

    grid_img.paste(sr_img_srresnet, (3 * margin + 2 * hr_img.width, margin))
    draw.text((3 * margin + 2 * hr_img.width, margin - 30), "SRResNet", fill="black", font=font)

    grid_img.paste(diff_bicubic, (4 * margin + 3 * hr_img.width, margin))
    draw.text((4 * margin + 3 * hr_img.width, margin - 30), "Bicubic Diff", fill="black", font=font)

    grid_img.paste(diff_sr, (5 * margin + 4 * hr_img.width, margin))
    draw.text((5 * margin + 4 * hr_img.width, margin - 30), "SRResNet Diff", fill="black", font=font)

    grid_img.show()

if __name__ == '__main__':
    visualize_sr_modified("img/butterfly.png")
