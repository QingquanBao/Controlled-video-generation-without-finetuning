import torch
import numpy as np
import os
import cv2
import sys
from PIL import Image
from torchvision import transforms


from ncut_pytorch import NCUT, rgb_from_tsne_3d
from ncut_pytorch.backbone import load_model, extract_features

from diffusers import StableDiffusionPipeline, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from matplotlib import pyplot as plt

from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((512, 512))])

img_dirs = 'golf_img'
images = []

# Iterate through all files in the directory
for filename in os.listdir(img_dirs):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(img_dirs, filename)
        img = Image.open(img_path).convert("RGB")

        # Apply transformation and append to list
        img_tensor = transform(img)
        images.append(img_tensor)

# Stack images along a new dimension to create a 4D tensor
images = torch.stack(images)

model = load_model(model_name="Diffusion(stabilityai/stable-diffusion-2)")

#images_ = torch.cat([images, images, images])
B_size = 4
model_features = {}
mode = model.to('cuda:0')
for i in tqdm(range(images.shape[0]//B_size)):
    start = i * B_size
    end = min((i + 1) * B_size, images.shape[0])
    batch = images[start:end]
    # model_features.extend(extract_features(images, model, node_type='up_3_resnets_1_conv1', layer=0, use_cuda=True))
    batch = batch.to('cuda:0')
    with torch.no_grad():
        model_features_of_batch = model(batch)
    if len(model_features.keys()) == 0:
        model_features = model_features_of_batch
    else:
        for key in model_features.keys():
            model_features[key].extend(model_features_of_batch[key])

for key in model_features.keys():
    model_features[key] = torch.stack(model_features[key]).cpu()
    model_features[key] = model_features[key].view(-1, *(model_features[key].size()[2:]))

for key in model_features.keys():
    print(key)

layers = [['latent']]
layers.append([key for key in model_features.keys() if 'attn' in key and 'attentions' in key])
layers.append([key for key in model_features.keys() if 'ff' in key and 'attentions' in key])
layers.append([key for key in model_features.keys() if 'block' in key and 'attentions' in key])
layers.append([key for key in model_features.keys() if 'conv' in key and 'resnets' in key])
layers.append([key for key in model_features.keys() if 'block' in key and 'resnets' in key])

layer_names = ['latent', 'attention.attn', 'attention.ff', 'attention.block', 'resnet.conv', 'resnet.block']

for l, layer in enumerate(layers):

    output_dir = f'feature_visualization/{layer_names[l]}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tsne_rgbs = []

    for layer_name in tqdm(layer):
        B_size, H, W, D = model_features[layer_name].shape
        inp = model_features[layer_name].reshape(-1, D)  # flatten
        eigvectors, eigvalues = NCUT(num_eig=10, device='cuda:0').fit_transform(inp)
        tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')

        eigvectors = eigvectors.reshape(B_size, H, W, 10)  # (B, H, W, num_eig)
        tsne_rgb = tsne_rgb.reshape(B_size, H, W, 3)  # (B, H, W, 3)
        tsne_rgbs.append(tsne_rgb)

    for idx in range(20):
        fig, axs = plt.subplots(1, len(layer)+1)
        axs[0].imshow(images[idx].permute(1,2,0))
        for i in range(len(layer)):
            axs[1+i].imshow(tsne_rgbs[i][idx])
        for ax in axs:
            ax.axis('off')
        plt.savefig(f'{output_dir}/{idx}', bbox_inches='tight', pad_inches=0, dpi=500)

    # Read images from the folder
    image_files = [img for img in os.listdir(output_dir) if img.endswith(".png")]
    image_files.sort()
    img_array = []
    for filename in image_files:
        img = cv2.imread(f'{output_dir}/' + filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    # Stack all images vertically
    out = cv2.vconcat(img_array)

    # Save the new big image
    cv2.imwrite(f'feature_visualization/{layer_names[l]}.png', out)

