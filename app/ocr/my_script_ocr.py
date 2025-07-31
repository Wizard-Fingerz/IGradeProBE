import requests
from django.conf import settings
import io


def extract_text_from_image(image_path):

    payload = {
        "data": image_path,  # Example data from the request
        "apiKey": settings.MYSCRIPT_API_KEY
    }

    with io.open(image_path, 'rb') as f:
        response = requests.post(
            'https://cloud.myscript.com/api/v4.0/iink/recognize', files={image_path: f}, data=payload)

    return response.content.decode()
