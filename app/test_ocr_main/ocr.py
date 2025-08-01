import pytesseract
from PIL import Image
import os

class TextRecognizer:
    def __init__(self, lang='eng'):
        self.lang = lang

    def recognize_texts(self, image_list):
        """
        Runs OCR on a list of cropped PIL images and returns extracted text.
        """
        results = []
        for i, img in enumerate(image_list):
            if isinstance(img, str) and os.path.exists(img):
                img = Image.open(img)

            text = pytesseract.image_to_string(img, lang=self.lang)
            results.append(text.strip())
        return results
