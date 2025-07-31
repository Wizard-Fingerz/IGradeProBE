import easyocr

def handwritten_to_text_easyocr(image_path, language='en'):
    """Converts handwritten text in an image to digital text using EasyOCR."""
    try:
        reader = easyocr.Reader([language])
        result = reader.readtext(image_path)

        # Extract text from the result
        extracted_text = ' '.join([text for _, text, _ in result])
        return extracted_text
    except Exception as e:
        return f"Error: {e}"

# Example usage
image_file = '525100527_502333_02.jpg'
extracted_text = handwritten_to_text_easyocr(image_file)
print(extracted_text)
