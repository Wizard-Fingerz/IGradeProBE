# recognizer.py
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TrOCRRecognizer:
    def __init__(self):
        print("[INFO] Loading TrOCR model and processor...")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    def recognize(self, cropped_images):
        recognized_texts = []
        for idx, img in enumerate(cropped_images):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            print(f"[INFO] Recognizing text in box {idx+1}...")
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            recognized_texts.append(generated_text)
        return recognized_texts
