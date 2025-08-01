# main.py
import cv2
from PIL import Image
from .detector import CRAFTTextDetector
from .recognizer import TrOCRRecognizer
import os



def load_image(image_path):
    # Return image path for detector, and cv2 image for cropping
    image = cv2.imread(image_path)
    return image_path, image

def save_cropped_images(image, boxes, output_dir="crops"):
    os.makedirs(output_dir, exist_ok=True)
    crops = []
    for idx, box in enumerate(boxes):
        # Skip invalid or malformed boxes
        if box is None or len(box) != 4:
            continue
        x_coords = [int(pt[0]) for pt in box]
        y_coords = [int(pt[1]) for pt in box]
        x_min, x_max = max(min(x_coords), 0), max(x_coords)
        y_min, y_max = max(min(y_coords), 0), max(y_coords)
        crop = image[y_min:y_max, x_min:x_max]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append(crop_pil)
        crop_path = os.path.join(output_dir, f"crop_{idx+1}.png")
        crop_pil.save(crop_path)
        print(f"[INFO] Saved cropped region to: {crop_path}")
    return crops


def extract_text_with_test_ocr(image_path):
    print("[INFO] Starting OCR pipeline...")

    # Step 1: Load Image
    img_path, image = load_image(image_path)
    print("[INFO] Image loaded.")

    # Step 2: Detect text boxes using CRAFT
    detector = CRAFTTextDetector()
    result = detector.detect(img_path)
    boxes = result['boxes']
    print(f"[INFO] Detected {len(boxes)} text regions.")

    # Step 3: Crop images
    cropped_images = save_cropped_images(image, boxes)

    # Step 4: Recognize text using TrOCR
    recognizer = TrOCRRecognizer()
    recognized_texts = recognizer.recognize(cropped_images)

    # Step 5: Print results
    print("\n=== Recognized Texts ===")
    for i, text in enumerate(recognized_texts, 1):
        print(f"{i}. {text}")



# if __name__ == "__main__":
#     image_path = "samples/booklet1.jpg"  # <-- change this to your test image
#     main(image_path)
    