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

    # Place the image labels in the grid
    hr_label.config(image=hr_photo)
    hr_label.grid(row=0, column=0, sticky='ew')
    bicubic_label.config(image=bicubic_photo)
    bicubic_label.grid(row=0, column=1, sticky='ew')
    srresnet_label.config(image=srresnet_photo)
    srresnet_label.grid(row=0, column=2, sticky='ew')

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

