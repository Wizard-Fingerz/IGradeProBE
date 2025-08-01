import cv2
import numpy as np
import torch
from craft_text_detector import Craft

class TextDetector:
    def __init__(self, cuda=False):
        self.craft = Craft(output_dir='outputs', crop_type='box', cuda=cuda)

    def detect_text(self, image_path):
        """
        Detects text regions in the given image and returns cropped image regions.
        """
        prediction_result = self.craft.detect_text(image_path)
        return prediction_result['boxes'], prediction_result['cropped_images']

    def release(self):
        """
        Free up resources.
        """
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()

class CRAFTTextDetector:
    def __init__(self, output_dir='output', crop_type='box', cuda=False):
        self.craft = Craft(output_dir=output_dir, crop_type=crop_type, cuda=cuda)

    def detect(self, image_path):
        return self.craft.detect_text(image_path)
