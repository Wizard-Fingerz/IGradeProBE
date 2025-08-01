# craft_detector.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os

from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import resize_aspect_ratio, normalizeMeanVariance

# Load the pretrained CRAFT model
def load_craft_model(model_path='models/craft_mlt_25k.pth', cuda=False):
    net = CRAFT()
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    if cuda:
        net = net.cuda()
    return net

# Run CRAFT detection
def detect_text_regions(net, image_path, canvas_size=1280, mag_ratio=1.5, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=False):
    image = Image.open(image_path).convert('RGB')
    img_np = np.array(image)

    # Resize and normalize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img_np, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, _ = net(x)

    # Get boxes from the score map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    boxes, _ = getDetBoxes(score_text, text_threshold, link_threshold, low_text, False)

    # Adjust coordinates back to original image
    boxes = np.array(boxes)
    boxes *= (ratio_w, ratio_h)

    return boxes.astype(int), img_np  # (boxes, original image)

# Crop boxes into separate image regions
def crop_text_regions(boxes, image):
    cropped_images = []
    for box in boxes:
        x_min = min([pt[0] for pt in box])
        y_min = min([pt[1] for pt in box])
        x_max = max([pt[0] for pt in box])
        y_max = max([pt[1] for pt in box])
        cropped = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped)
    return cropped_images
