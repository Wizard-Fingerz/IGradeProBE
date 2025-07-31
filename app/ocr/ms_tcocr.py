import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import logging
import os


# ---------------------------------------------
# Load TrOCR model and processor
# ---------------------------------------------
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------
# Function: Preprocess image
# ---------------------------------------------
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Convert back to RGB for PIL
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb)
    
    # Debug: Save preprocessed image
    debug_path = "preprocessed_" + os.path.basename(image_path)
    pil_image.save(debug_path)
    logging.info(f"Saved preprocessed image to {debug_path}")
    
    return pil_image


# ---------------------------------------------
# Function: Process full page with TrOCR
# ---------------------------------------------
def process_full_page(image_path):
    # Load and preprocess image
    image = preprocess_image(image_path)
    
    # Debug: Show image dimensions
    logging.info(f"Image size: {image.size}")
    
    # Process with TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate with more parameters for better results
    generated_ids = model.generate(
        pixel_values,
        max_length=512,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text.strip()


# ---------------------------------------------
# Function: Detect lines from OCR output
# ---------------------------------------------
def detect_lines_from_ocr(text):
    # Split text into lines based on natural line breaks
    lines = text.split('\n')
    
    # Filter out empty lines and clean up
    lines = [line.strip() for line in lines if line.strip()]
    
    return lines


# ---------------------------------------------
# Function: Parse structured content from OCR text
# ---------------------------------------------
def extract_all_text_between_as_ae(text):
    if not isinstance(text, str):
        print(f"Warning: Received non-string input ({type(text)})")
        return []

    main_question_pattern = r'MQS(.*?)MQE'
    question_pattern = r'QS(.*?)QE'
    answer_pattern = r'AS(.*?)AE'

    main_questions = re.findall(main_question_pattern, text, re.DOTALL)
    questions = re.findall(question_pattern, text, re.DOTALL)
    answers = re.findall(answer_pattern, text, re.DOTALL)

    main_questions = [mq.replace("\\n", "\n").strip() for mq in main_questions]
    questions = [q.replace("\\n", "\n").strip() for q in questions]
    answers = [a.replace("\\n", "\n").strip() for a in answers]

    result = []

    for mq in main_questions:
        result.append({"main_question": mq})

    for i in range(max(len(questions), len(answers))):
        entry = {}
        if i < len(questions):
            entry["question"] = questions[i]
        if i < len(answers):
            entry["answer"] = answers[i]
        result.append(entry)

    return result


# ---------------------------------------------
# Main function
# ---------------------------------------------
def process_handwritten_image(image_path):
    logging.info(f"Processing image: {image_path}")
    
    # Process full page
    full_text = process_full_page(image_path)
    logging.info("Full extracted text:")
    logging.info(full_text)
    
    # Detect lines from OCR output
    lines = detect_lines_from_ocr(full_text)
    logging.info("Detected lines:")
    for i, line in enumerate(lines):
        logging.info(f"Line {i+1}: {line}")
    
    # Extract structured data
    structured_data = extract_all_text_between_as_ae(full_text)
    logging.info("Structured data:")
    logging.info(structured_data)
    return structured_data


# ---------------------------------------------
# Run it
# ---------------------------------------------
if __name__ == "__main__":
    # Replace this with your actual image path
    image_path = "525100527_502333_02.jpg"
    result = process_handwritten_image(image_path)
    print("\nFinal Structured Output:\n", result)
