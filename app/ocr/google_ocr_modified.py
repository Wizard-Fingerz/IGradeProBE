import logging
from difflib import get_close_matches
import io
from google.cloud import vision
from google.oauth2 import service_account
import re

from app.questions.models import SubjectQuestion


def detect_document_modified(image_path, json_path):

    # Set up the Google Cloud Vision client
    credentials = service_account.Credentials.from_service_account_file(
        json_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    extracted_text = response.full_text_annotation.text
    # print(TextOutput)
    return extracted_text


logging.basicConfig(level=logging.INFO)


def detect_document_modified(image_path, json_path):
    logging.info(f"Starting document text detection for {image_path}")

    # Set up the Google Cloud Vision client
    try:
        credentials = service_account.Credentials.from_service_account_file(
            json_path)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        # logging.info("Google Cloud Vision client set up successfully")
    except Exception as e:
        # logging.error(f"Failed to set up Google Cloud Vision client: {str(e)}")
        return None

    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        # logging.info("Image file read successfully")
    except Exception as e:
        # logging.error(f"Failed to read image file: {str(e)}")
        return None

    image = vision.Image(content=content)

    try:
        response = client.document_text_detection(image=image)
        # logging.info("Document text detection successful")
    except Exception as e:
        # logging.error(f"Failed to detect document text: {str(e)}")
        return None

    try:
        extracted_text = response.full_text_annotation.text
        # logging.info("Extracted text successfully")
        return extracted_text
    except Exception as e:
        # logging.error(f"Failed to extract text: {str(e)}")
        return None


def extract_all_text_between_as_ae(text):
    if not isinstance(text, str):  # Ensure text is a string
        print(f"Warning: extract_all_text_between_as_ae received non-string input ({type(text)})")
        return []  # Return an empty list to prevent errors

    # Define patterns for main questions, questions, and answers
    main_question_pattern = r'MQS(.*?)MQE'
    question_pattern = r'QS(.*?)QE'
    answer_pattern = r'AS(.*?)AE'

    # Find all matches for main questions, questions, and answers
    main_questions = re.findall(main_question_pattern, text, re.DOTALL)
    questions = re.findall(question_pattern, text, re.DOTALL)
    answers = re.findall(answer_pattern, text, re.DOTALL)

    # Clean up the matches by replacing \n with actual new lines and stripping whitespace
    main_questions = [mq.replace("\\n", "\n").strip() for mq in main_questions]
    questions = [q.replace("\\n", "\n").strip() for q in questions]
    answers = [a.replace("\\n", "\n").strip() for a in answers]

    # Combine main questions, questions, and answers into a structured format
    result = []

    # Add main questions to the result
    for mq in main_questions:
        result.append({"main_question": mq})

    # Add questions and answers to the result
    for i in range(max(len(questions), len(answers))):
        entry = {}
        if i < len(questions):
            entry["question"] = questions[i]
        if i < len(answers):
            entry["answer"] = answers[i]
        result.append(entry)

    return result


def find_matching_question(extracted_question):
    all_questions = SubjectQuestion.objects.values_list("question", flat=True)
    best_match = get_close_matches(
        extracted_question, all_questions, n=1, cutoff=0.6)

    if best_match:
        return SubjectQuestion.objects.get(question=best_match[0])
    return None  # No good match found
