import cv2
import numpy as np
from typing import List, Tuple

def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws bounding boxes on the image.

    Args:
        image (np.ndarray): Original image.
        boxes (List[Tuple[int, int, int, int]]): List of bounding boxes.
        color (Tuple[int, int, int]): Box color.
        thickness (int): Thickness of lines.

    Returns:
        np.ndarray: Image with boxes drawn.
    """
    image_copy = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy

def sort_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Sort text boxes top-to-bottom, left-to-right.

    Args:
        boxes (List[Tuple[int, int, int, int]]): List of bounding boxes.

    Returns:
        List[Tuple[int, int, int, int]]: Sorted boxes.
    """
    return sorted(boxes, key=lambda box: (box[1], box[0]))

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy (grayscale + threshold).

    Args:
        image (np.ndarray): Original image.

    Returns:
        np.ndarray: Preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
