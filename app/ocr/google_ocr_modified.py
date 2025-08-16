import logging
from difflib import get_close_matches
import io
from google.cloud import vision
from google.oauth2 import service_account
import re

from app.questions.models import SubjectQuestion

logging.basicConfig(level=logging.INFO)



def detect_document_modified(image_path, json_path):
    logging.info(f"Starting document text detection for {image_path}")

    # Set up the Google Cloud Vision client
    try:
        credentials = service_account.Credentials.from_service_account_file(
            json_path)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        logging.info("Google Cloud Vision client set up successfully")
    except Exception as e:
        logging.error(f"Failed to set up Google Cloud Vision client: {str(e)}")
        return None

    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        logging.info("Image file read successfully")
    except Exception as e:
        logging.error(f"Failed to read image file: {str(e)}")
        return None

    image = vision.Image(content=content)

    try:
        response = client.document_text_detection(image=image)
        logging.info("Document text detection successful")
    except Exception as e:
        logging.error(f"Failed to detect document text: {str(e)}")
        return None

    try:
        extracted_text = response.full_text_annotation.text
        logging.info("Extracted text successfully")
        return extracted_text
    except Exception as e:
        logging.error(f"Failed to extract text: {str(e)}")
        return None

import re

# $MQ$ ... $/MQ$
# $Q$ ... $/Q$
# $A$ ... $/A$

# pattern = r'\$MQ\$(.*?)\$/MQ\$|\$Q\$(.*?)\$/Q\$|\$A\$(.*?)\$/A\$'

# main_question_pattern = r'@\$MQ\$@(.*?)@\$MQ_END\$@'
# question_pattern = r'@\$Q\$@(.*?)@\$Q_END\$@'
# answer_pattern = r'@\$A\$@(.*?)@\$A_END\$@'


def extract_all_text_sequentially(text):
    """
    Extracts main questions, questions, and answers from text using regex.
    If an ending marker is missing, the function will break to a new question/answer
    whenever it finds the beginning marker of the next question/answer.
    If the ending marker is present, it will use it as the boundary.
    """
    if not isinstance(text, str):
        print(f"Warning: extract_all_text_sequentially received non-string input ({type(text)})")
        return []

    # Define start and end markers for each type
    MQ_START = r'[\$S](?:START|SMART)'
    MQ_END = r'[\$S]STOP'
    Q_START = r'[\$S]BEGIN'
    Q_END = r'[\$S]{2}END'
    A_START = r'[\$S]INIT'
    A_END = r'[\$S]{2}HALT'

    # Build a regex that matches any of the start markers, capturing the marker and its position
    start_pattern = re.compile(
        rf'({MQ_START})|({Q_START})|({A_START})',
        re.IGNORECASE
    )

    # Helper to find the next start marker after a given index
    def find_next_start(text, start_idx):
        match = start_pattern.search(text, start_idx)
        return match.start() if match else len(text)

    result = []
    current = {}

    idx = 0
    text_len = len(text)
    while idx < text_len:
        # Search for the next start marker
        match = start_pattern.search(text, idx)
        if not match:
            break
        marker = match.group(0)
        start_idx = match.end()

        # Determine which type of marker it is and set the appropriate end marker
        if re.match(MQ_START, marker, re.IGNORECASE):
            end_regex = re.compile(MQ_END, re.IGNORECASE)
            key = "main_question"
        elif re.match(Q_START, marker, re.IGNORECASE):
            end_regex = re.compile(Q_END, re.IGNORECASE)
            key = "question"
        elif re.match(A_START, marker, re.IGNORECASE):
            end_regex = re.compile(A_END, re.IGNORECASE)
            key = "answer"
        else:
            # Should not happen
            idx = start_idx
            continue

        # Try to find the ending marker for this section
        end_match = end_regex.search(text, start_idx)
        next_start = find_next_start(text, start_idx)
        if end_match and end_match.start() < next_start:
            # Ending marker found before next start marker
            content_end = end_match.start()
            next_idx = end_match.end()
        else:
            # No ending marker, or next start marker comes first
            content_end = next_start
            next_idx = next_start

        content = text[start_idx:content_end].strip()

        # If starting a new main_question or question, flush current if needed
        if key == "main_question":
            if current:
                result.append(current)
                current = {}
            current[key] = content
        elif key == "question":
            if current and "question" in current:
                result.append(current)
                current = {}
            current[key] = content
        elif key == "answer":
            current[key] = content
            result.append(current)
            current = {}

        idx = next_idx

    if current:
        result.append(current)

    return result


def extract_all_text_between_as_ae(text):
    if not isinstance(text, str):  # Ensure text is a string
        print(text)
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


# def find_matching_question(extracted_question):
#     all_questions = SubjectQuestion.objects.values_list("question", flat=True)
#     print(all_questions)
#     best_match = get_close_matches(
#         extracted_question, all_questions, n=1, cutoff=0.6)

#     if best_match:
#         return SubjectQuestion.objects.filter(question=best_match[0]).first()
#     return None  # No good match found

from difflib import get_close_matches

def find_matching_question(extracted_question):
    # Get only parent questions (parent_question is None)
    parent_questions = SubjectQuestion.objects.filter(parent_question__isnull=True).values_list("question", flat=True)
    
    best_match = get_close_matches(extracted_question, parent_questions, n=1, cutoff=0.9)

    if best_match:
        # Return the matched parent question object
        return SubjectQuestion.objects.filter(question=best_match[0], parent_question__isnull=True).first()

    # No good match in parents, now try subquestions
    sub_questions = SubjectQuestion.objects.filter(parent_question__isnull=False).values_list("question", flat=True)
    best_match_sub = get_close_matches(extracted_question, sub_questions, n=1, cutoff=0.9)

    if best_match_sub:
        return SubjectQuestion.objects.filter(question=best_match_sub[0], parent_question__isnull=False).first()

    return None  # No good match found
