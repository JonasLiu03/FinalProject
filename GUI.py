# # import tkinter as tk
# # from tkinter import filedialog
# # from PIL import ImageTk
# # from utils import *
# # from PIL import Image
# #
# # root = tk.Tk()
# # root.title("Image Super-Resolution")
# # root.geometry("1200x400")
# #
# #
# # root.grid_columnconfigure(0, weight=1)
# # root.grid_rowconfigure(1, weight=1)
# #
# # title_label = tk.Label(root, text="Image Super Resolution", font=("Arial", 24))
# # title_label.grid(row=0, column=0, sticky='ew')
# #
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # srresnet_checkpoint = "SrresnetCheckpoint(cropsize96batch16prelu)/checkpoint_epoch_43_srresnet.pth.tar"
# # srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# # srresnet.eval()
# #
# # hr_photo = None
# # bicubic_photo = None
# # srresnet_photo = None
# # hr_label = None
# # bicubic_label = None
# # srresnet_label = None
# # frame_images = None
# # def visualize_sr(img, halve=False):
# #     hr_img = Image.open(img).convert('RGB')
# #     if halve:
# #         hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
# #     lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)
# #     bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)
# #     sr_img_srresnet = srresnet(convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device))
# #     sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
# #     sr_img_srresnet = convert_image(sr_img_srresnet, '[-1, 1]', 'pil')
# #     return hr_img, bicubic_img, sr_img_srresnet
# #
# #
# #
# #
# # def display_images(hr_image, bicubic_image, srresnet_image):
# #     global hr_photo, bicubic_photo, srresnet_photo, hr_label, bicubic_label, srresnet_label, frame_images
# #     max_width = root.winfo_width() // 3
# #     max_height = root.winfo_height() - 100
# #
# #     # Resize the images to fit within the new dimensions
# #     hr_image = hr_image.resize((max_width, max_height), Image.ANTIALIAS)
# #     bicubic_image = bicubic_image.resize((max_width, max_height), Image.ANTIALIAS)
# #     srresnet_image = srresnet_image.resize((max_width, max_height), Image.ANTIALIAS)
# #
# #     # Convert the PIL images to Tkinter PhotoImage format
# #     hr_photo = ImageTk.PhotoImage(hr_image)
# #     bicubic_photo = ImageTk.PhotoImage(bicubic_image)
# #     srresnet_photo = ImageTk.PhotoImage(srresnet_image)
# #
# #     # Create frames and labels if they don't exist
# #     if frame_images is None:
# #         frame_images = tk.Frame(root)
# #         frame_images.grid(row=1, column=0, sticky='ew', padx=10, pady=10)
# #         hr_label = tk.Label(frame_images, image=hr_photo)
# #         bicubic_label = tk.Label(frame_images, image=bicubic_photo)
# #         srresnet_label = tk.Label(frame_images, image=srresnet_photo)
# #
# #     # Place the image labels in the grid
# #     hr_label.config(image=hr_photo)
# #     hr_label.grid(row=0, column=0, sticky='ew')
# #     bicubic_label.config(image=bicubic_photo)
# #     bicubic_label.grid(row=0, column=1, sticky='ew')
# #     srresnet_label.config(image=srresnet_photo)
# #     srresnet_label.grid(row=0, column=2, sticky='ew')
# #
# #     # Keep a reference to the images to prevent garbage collection
# #     hr_label.image = hr_photo
# #     bicubic_label.image = bicubic_photo
# #     srresnet_label.image = srresnet_photo
# #
# #
# # def resize_images(event):
# #     global hr_label, bicubic_label, srresnet_label, hr_photo, bicubic_photo, srresnet_photo
# #     # Ensure the labels are created before attempting to resize
# #     if hr_label and bicubic_label and srresnet_label:
# #         new_width = event.width // 3
# #         new_height = event.height // 3  # Adjust as needed
# #
# #         # Update the images if they have been loaded
# #         if hr_photo and bicubic_photo and srresnet_photo:
# #             hr_photo = ImageTk.PhotoImage(hr_image.resize((new_width, new_height), Image.ANTIALIAS))
# #             bicubic_photo = ImageTk.PhotoImage(bicubic_image.resize((new_width, new_height), Image.ANTIALIAS))
# #             srresnet_photo = ImageTk.PhotoImage(srresnet_image.resize((new_width, new_height), Image.ANTIALIAS))
# #
# #             hr_label.config(image=hr_photo)
# #             bicubic_label.config(image=bicubic_photo)
# #             srresnet_label.config(image=srresnet_photo)
# #
# #             # Keep a reference to the images to prevent garbage collection
# #             hr_label.image = hr_photo
# #             bicubic_label.image = bicubic_photo
# #             srresnet_label.image = srresnet_photo
# #
# # def open_file():
# #     global hr_image, bicubic_image, srresnet_image
# #     file_path = filedialog.askopenfilename()
# #     if file_path:
# #         # Load the images and update the global variables
# #         hr_image, bicubic_image, srresnet_image = visualize_sr(file_path)
# #         display_images(hr_image, bicubic_image, srresnet_image)
# #
# # root.bind('<Configure>', display_images)
# # change_button = tk.Button(root, text="Choose Image", command=open_file)
# # change_button.grid(row=2, column=0, pady=20)
# #
# # root.mainloop()
# #
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
#
# # 使用正确的导入和初始化方法
# root = tk.Tk()
# root.title("Image Super-Resolution")
# root.geometry("1200x600")
#
# # 设备和模型初始化
# device = 'cpu'  # 示例使用CPU
# model = None  # 这里应初始化模型
#
# def load_model():
#     global model
#     model_path = "path_to_model.pth"  # 设置模型路径
#     # 模型加载逻辑
#     model = "Loaded model"  # 假设模型已加载
#
# def visualize_sr(file_path):
#     # 图像处理和超分辨率逻辑
#     hr_image = Image.open(file_path)  # 示例使用PIL加载图片
#     lr_image = hr_image.resize((hr_image.width // 2, hr_image.height // 2), Image.LANCZOS)
#     sr_image = lr_image  # 假设已应用超分辨率
#     return hr_image, lr_image, sr_image
#
# def update_display():
#     global hr_photo, lr_photo, sr_photo
#     hr_photo = ImageTk.PhotoImage(hr_image)
#     lr_photo = ImageTk.PhotoImage(lr_image)
#     sr_photo = ImageTk.PhotoImage(sr_image)
#
#     hr_label.config(image=hr_photo)
#     lr_label.config(image=lr_photo)
#     sr_label.config(image=sr_photo)
#
# def open_file():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         global hr_image, lr_image, sr_image
#         hr_image, lr_image, sr_image = visualize_sr(file_path)
#         update_display()
#
# load_model()  # 加载模型
#
# # UI组件初始化
# hr_label = tk.Label(root)
# lr_label = tk.Label(root)
# sr_label = tk.Label(root)
#
# hr_label.pack(side="left")
# lr_label.pack(side="left")
# sr_label.pack(side="left")
#
# button = tk.Button(root, text="Open File", command=open_file)
# button.pack(side="bottom")
#
# root.mainloop()
#
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from utils import *
from PIL import Image

root = tk.Tk()
root.title("Image Super-Resolution")
root.geometry("1200x400")


root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

title_label = tk.Label(root, text="Image Super Resolution", font=("Arial", 24))
title_label.grid(row=0, column=0, sticky='ew')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet_checkpoint = "SrresnetCheckpoint/checkpoint_epoch_43_srresnet.pth.tar"
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()

hr_photo = None
bicubic_photo = None
srresnet_photo = None
hr_label = None
bicubic_label = None
srresnet_label = None
frame_images = None
def visualize_sr(img, halve=False):
    hr_img = Image.open(img).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)
    bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)
    sr_img_srresnet = srresnet(convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, '[-1, 1]', 'pil')
    return hr_img, bicubic_img, sr_img_srresnet




def display_images(hr_image, bicubic_image, srresnet_image):
    global hr_photo, bicubic_photo, srresnet_photo, hr_label, bicubic_label, srresnet_label, frame_images
    max_width = root.winfo_width() // 3
    max_height = root.winfo_height() - 100

    # Resize the images to fit within the new dimensions
    hr_image = hr_image.resize((max_width, max_height), Image.ANTIALIAS)
    bicubic_image = bicubic_image.resize((max_width, max_height), Image.ANTIALIAS)
    srresnet_image = srresnet_image.resize((max_width, max_height), Image.ANTIALIAS)

    # Convert the PIL images to Tkinter PhotoImage format
    hr_photo = ImageTk.PhotoImage(hr_image)
    bicubic_photo = ImageTk.PhotoImage(bicubic_image)
    srresnet_photo = ImageTk.PhotoImage(srresnet_image)

    # Create frames and labels if they don't exist
    if frame_images is None:
        frame_images = tk.Frame(root)
        frame_images.grid(row=1, column=0, sticky='ew', padx=10, pady=10)

        hr_label = tk.Label(frame_images, image=hr_photo)
        bicubic_label = tk.Label(frame_images, image=bicubic_photo)
        srresnet_label = tk.Label(frame_images, image=srresnet_photo)

    # Adding title labels
    hr_title = tk.Label(frame_images, text="HR Image", font=("Arial", 14))
    bicubic_title = tk.Label(frame_images, text="Bicubic Image", font=("Arial", 14))
    srresnet_title = tk.Label(frame_images, text="Srresnet Image", font=("Arial", 14))

    # Place the title labels in the grid
    hr_title.grid(row=0, column=0, sticky='ew')
    bicubic_title.grid(row=0, column=1, sticky='ew')
    srresnet_title.grid(row=0, column=2, sticky='ew')

    # Place the image labels in the grid
    hr_label.config(image=hr_photo)
    hr_label.grid(row=1, column=0, sticky='ew')
    bicubic_label.config(image=bicubic_photo)
    bicubic_label.grid(row=1, column=1, sticky='ew')
    srresnet_label.config(image=srresnet_photo)
    srresnet_label.grid(row=1, column=2, sticky='ew')

    # Keep a reference to the images to prevent garbage collection
    hr_label.image = hr_photo
    bicubic_label.image = bicubic_photo
    srresnet_label.image = srresnet_photo
def resize_images(event):
    global hr_label, bicubic_label, srresnet_label, hr_photo, bicubic_photo, srresnet_photo
    # Ensure the labels are created before attempting to resize
    if hr_label and bicubic_label and srresnet_label:
        new_width = event.width // 3
        new_height = event.height // 3  # Adjust as needed

        # Update the images if they have been loaded
        if hr_photo and bicubic_photo and srresnet_photo:
            hr_photo = ImageTk.PhotoImage(hr_image.resize((new_width, new_height), Image.ANTIALIAS))
            bicubic_photo = ImageTk.PhotoImage(bicubic_image.resize((new_width, new_height), Image.ANTIALIAS))
            srresnet_photo = ImageTk.PhotoImage(srresnet_image.resize((new_width, new_height), Image.ANTIALIAS))

            hr_label.config(image=hr_photo)
            bicubic_label.config(image=bicubic_photo)
            srresnet_label.config(image=srresnet_photo)

            # Keep a reference to the images to prevent garbage collection
            hr_label.image = hr_photo
            bicubic_label.image = bicubic_photo
            srresnet_label.image = srresnet_photo

def open_file():
    global hr_image, bicubic_image, srresnet_image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the images and update the global variables
        hr_image, bicubic_image, srresnet_image = visualize_sr(file_path)
        display_images(hr_image, bicubic_image, srresnet_image)

root.bind('<Configure>', display_images)
change_button = tk.Button(root, text="Choose Image", command=open_file)
change_button.grid(row=2, column=0, pady=20)

root.mainloop()

