import io
from google.cloud import vision
from google.oauth2 import service_account


def detect_document_modified(image_path, json_path):
    
    # Set up the Google Cloud Vision client
    credentials = service_account.Credentials.from_service_account_file(json_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    extracted_text = response.full_text_annotation.text
    # print(TextOutput)
    return extracted_text

