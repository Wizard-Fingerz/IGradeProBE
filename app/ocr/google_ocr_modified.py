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


# def extract_all_text_sequentially(text):
#     """
#     Extracts main questions, questions, and answers from text using regex.
#     If an ending marker is missing, the function will break to a new question/answer
#     whenever it finds the beginning marker of the next question/answer.
#     If the ending marker is present, it will use it as the boundary.
#     """
#     if not isinstance(text, str):
#         print(f"Warning: extract_all_text_sequentially received non-string input ({type(text)})")
#         return []

#     # Define start and end markers for each type
#     MQ_START = r'[\$S](?:START|SMART)'
#     MQ_END = r'[\$S]STOP'
#     Q_START = r'[\$S]BEGIN'
#     Q_END = r'[\$S]{2}END'
#     A_START = r'[\$S]INIT'
#     A_END = r'[\$S]{2}HALT'

#     # Build a regex that matches any of the start markers, capturing the marker and its position
#     start_pattern = re.compile(
#         rf'({MQ_START})|({Q_START})|({A_START})',
#         re.IGNORECASE
#     )

#     # Helper to find the next start marker after a given index
#     def find_next_start(text, start_idx):
#         match = start_pattern.search(text, start_idx)
#         return match.start() if match else len(text)

#     result = []
#     current = {}

#     idx = 0
#     text_len = len(text)
#     while idx < text_len:
#         # Search for the next start marker
#         match = start_pattern.search(text, idx)
#         if not match:
#             break
#         marker = match.group(0)
#         start_idx = match.end()

#         # Determine which type of marker it is and set the appropriate end marker
#         if re.match(MQ_START, marker, re.IGNORECASE):
#             end_regex = re.compile(MQ_END, re.IGNORECASE)
#             key = "main_question"
#         elif re.match(Q_START, marker, re.IGNORECASE):
#             end_regex = re.compile(Q_END, re.IGNORECASE)
#             key = "question"
#         elif re.match(A_START, marker, re.IGNORECASE):
#             end_regex = re.compile(A_END, re.IGNORECASE)
#             key = "answer"
#         else:
#             # Should not happen
#             idx = start_idx
#             continue

#         # Try to find the ending marker for this section
#         end_match = end_regex.search(text, start_idx)
#         next_start = find_next_start(text, start_idx)
#         if end_match and end_match.start() < next_start:
#             # Ending marker found before next start marker
#             content_end = end_match.start()
#             next_idx = end_match.end()
#         else:
#             # No ending marker, or next start marker comes first
#             content_end = next_start
#             next_idx = next_start

#         content = text[start_idx:content_end].strip()

#         # If starting a new main_question or question, flush current if needed
#         if key == "main_question":
#             if current:
#                 result.append(current)
#                 current = {}
#             current[key] = content
#         elif key == "question":
#             if current and "question" in current:
#                 result.append(current)
#                 current = {}
#             current[key] = content
#         elif key == "answer":
#             current[key] = content
#             result.append(current)
#             current = {}

#         idx = next_idx

#     if current:
#         result.append(current)

#     return result



import re
import re

def clean_final_output(data):
    """
    Cleans up question/answer dicts by removing regex tags, exam marks,
    and common exam artifacts like 'Index Number' or 'Do not write in this margin'.
    """
    cleaned = []
    # Patterns for unwanted exam artifacts
    artifact_patterns = [
        r'Index\s*Number.*?(?=\.|$)',  # Remove "Index Number..." up to period or end
        r'Do not write in this margin.*?(?=\.|$)',  # Remove "Do not write in this margin..." up to period or end
        r'Name\s*.*?(?=\.|$)',  # Remove "Name ..." up to period or end
        r'Candidate\s*Number.*?(?=\.|$)',  # Remove "Candidate Number..." up to period or end
        r'Centre\s*Number.*?(?=\.|$)',  # Remove "Centre Number..." up to period or end
        r'For\s*Examiner.*?(?=\.|$)',  # Remove "For Examiner..." up to period or end
        r'For\s*Marker.*?(?=\.|$)',  # Remove "For Marker..." up to period or end
        r'Page\s*\d+\s*of\s*\d+',  # Remove "Page x of y"
        r'Question\s*Paper.*?(?=\.|$)',  # Remove "Question Paper..." up to period or end
        r'Instructions.*?(?=\.|$)',  # Remove "Instructions..." up to period or end
        r'Write your name.*?(?=\.|$)',  # Remove "Write your name..." up to period or end
        r'Write your index number.*?(?=\.|$)',  # Remove "Write your index number..." up to period or end
        r'Write in black or blue ink.*?(?=\.|$)',  # Remove "Write in black or blue ink..." up to period or end
        r'All questions should be answered.*?(?=\.|$)',  # Remove "All questions should be answered..." up to period or end
        r'Answer all questions.*?(?=\.|$)',  # Remove "Answer all questions..." up to period or end
        r'Each question carries.*?(?=\.|$)',  # Remove "Each question carries..." up to period or end
        r'You may use.*?(?=\.|$)',  # Remove "You may use..." up to period or end
        r'You are reminded.*?(?=\.|$)',  # Remove "You are reminded..." up to period or end
        r'Additional materials.*?(?=\.|$)',  # Remove "Additional materials..." up to period or end
        r'Blank page',  # Remove "Blank page"
    ]

    artifact_regex = re.compile("|".join(artifact_patterns), re.IGNORECASE)

    for item in data:
        q = item.get("question", "")
        a = item.get("answer", "")

        # Remove all parsing regex leftovers
        q = re.sub(r'(\$\$END|\$INIT|\$BEGIN|\$STOP|\$HALT)', '', q)
        a = re.sub(r'(\$\$END|\$INIT|\$BEGIN|\$STOP|\$HALT)', '', a)

        # Remove question numbering artifacts like "11:" at start
        q = re.sub(r'^\s*\d+\s*[:.)-]?\s*', '', q)
        a = re.sub(r'^\s*\d+\s*[:.)-]?\s*', '', a)

        # Remove marks like [1 mark], (2 marks), [15 marks] etc.
        q = re.sub(r'\[\s*\d+\s*marks?\s*\]', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\(\s*\d+\s*marks?\s*\)', '', q, flags=re.IGNORECASE)
        a = re.sub(r'\[\s*\d+\s*marks?\s*\]', '', a, flags=re.IGNORECASE)
        a = re.sub(r'\(\s*\d+\s*marks?\s*\)', '', a, flags=re.IGNORECASE)

        # Remove common exam artifacts from questions
        q = artifact_regex.sub('', q)

        # Remove extra spaces & normalize newlines
        q = re.sub(r'\s+', ' ', q).strip()
        a = re.sub(r'\s+', ' ', a).strip()

        cleaned.append({"question": q, "answer": a})
    return cleaned


def clean_text(text: str) -> str:
    if not text:
        return text
    # remove marks like [1 mark], [2 marks], etc
    text = re.sub(r"\[\d+\s*marks?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\s*mark\]", "", text, flags=re.IGNORECASE)

    # remove leftover $$BEGIN, $$END, $INIT, $HALT, etc
    text = re.sub(r"\$\$?[A-Z]+\b", "", text)

    # strip extra spaces & newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_all_text_sequentially(text):
    """
    Improved fault-tolerant extraction:
    - Supports questions spilling after $$END (e.g., "11: Give four reasons ...").
    - Uses markers when available.
    - Recovers missing questions by scanning unmapped text for question-like phrases.
    - Ensures every answer has a mapped question (or a placeholder).
    - Cleans up common exam artifacts and instructions from questions.
    """

    if not isinstance(text, str):
        print(f"Warning: extract_all_text_sequentially received non-string input ({type(text)})")
        return []

    # Define markers
    MQ_START = r'[\$S](?:START|SMART)'
    MQ_END   = r'[\$S]STOP'
    Q_START  = r'[\$S]BEGIN'
    Q_END    = r'[\$S]{2}END'
    A_START  = r'[\$S]INIT'
    A_END    = r'[\$S]{2}HALT'

    start_pattern = re.compile(
        rf'({MQ_START})|({Q_START})|({A_START})',
        re.IGNORECASE
    )

    def find_next_start(text, start_idx):
        match = start_pattern.search(text, start_idx)
        return match.start() if match else len(text)

    result = []
    current = {}
    idx = 0
    text_len = len(text)

    mapped_ranges = []  # track extracted regions for recovery later

    # regex to detect question-like text
    question_like = re.compile(
        r'(^\d+[:.)])|(who|what|when|where|why|how|explain|describe|define|list|state|give)',
        re.IGNORECASE
    )

    # Patterns to clean up from questions (and optionally answers)
    question_artifact_patterns = [
        r'Index Number:.*?(?=\n|$)',  # Remove "Index Number: ..." starting from the colon up to newline or end
        r'Candidate Name\s*:?[\s\S]*?(?=\n|$)',  # Remove "Candidate Name: ..." up to newline or end
        r'Do not write in this margin[\s\S]*?(?=\n|$)',  # Remove "Do not write in this margin..." up to newline or end
        r'Do not write on this page[\s\S]*?(?=\n|$)',  # Remove "Do not write on this page..." up to newline or end
        r'For Examiner\'?s? Use Only[\s\S]*?(?=\n|$)',  # Remove "For Examiner's Use Only..." up to newline or end
        r'You may use.*?(?=\.|$)',  # Remove "You may use..." up to period or end
        r'You are reminded.*?(?=\.|$)',  # Remove "You are reminded..." up to period or end
        r'Additional materials.*?(?=\.|$)',  # Remove "Additional materials..." up to period or end
        r'Blank page',  # Remove "Blank page"
        r'Write your answers.*?(?=\.|$)',  # Remove "Write your answers..." up to period or end
        r'Answer all questions.*?(?=\.|$)',  # Remove "Answer all questions..." up to period or end
        r'All questions carry equal marks.*?(?=\.|$)',  # Remove "All questions carry equal marks..." up to period or end
        r'Each question carries.*?(?=\.|$)',  # Remove "Each question carries..." up to period or end
        r'Use a separate answer sheet.*?(?=\.|$)',  # Remove "Use a separate answer sheet..." up to period or end
        r'Instructions.*?(?=\.|$)',  # Remove "Instructions..." up to period or end
        r'Page \d+ of \d+',  # Remove "Page x of y"
        r'Centre Number\s*:?[\s\S]*?(?=\n|$)',  # Remove "Centre Number: ..." up to newline or end
        r'School Name\s*:?[\s\S]*?(?=\n|$)',  # Remove "School Name: ..." up to newline or end
        r'Class\s*:?[\s\S]*?(?=\n|$)',  # Remove "Class: ..." up to newline or end
        r'Subject\s*:?[\s\S]*?(?=\n|$)',  # Remove "Subject: ..." up to newline or end
        r'Candidate No\.?\s*:?[\s\S]*?(?=\n|$)',  # Remove "Candidate No: ..." up to newline or end
        r'Examiner.*?(?=\.|$)',  # Remove "Examiner..." up to period or end
        r'Invigilator.*?(?=\.|$)',  # Remove "Invigilator..." up to period or end
        r'Write your name.*?(?=\.|$)',  # Remove "Write your name..." up to period or end
        r'Write your index number.*?(?=\.|$)',  # Remove "Write your index number..." up to period or end
        r'Write in black or blue ink.*?(?=\.|$)',  # Remove "Write in black or blue ink..." up to period or end
        r'Write legibly.*?(?=\.|$)',  # Remove "Write legibly..." up to period or end
        r'Fill in the boxes.*?(?=\.|$)',  # Remove "Fill in the boxes..." up to period or end
        r'Check that this question paper.*?(?=\.|$)',  # Remove "Check that this question paper..." up to period or end
        r'You must not use.*?(?=\.|$)',  # Remove "You must not use..." up to period or end
        r'You must answer on the question paper.*?(?=\.|$)',  # Remove "You must answer on the question paper..." up to period or end
        r'You must answer on the answer booklet.*?(?=\.|$)',  # Remove "You must answer on the answer booklet..." up to period or end
        r'You must answer on the separate answer sheet.*?(?=\.|$)',  # Remove "You must answer on the separate answer sheet..." up to period or end
        r'You must answer all questions.*?(?=\.|$)',  # Remove "You must answer all questions..." up to period or end
        r'You must answer only one question.*?(?=\.|$)',  # Remove "You must answer only one question..." up to period or end
        r'You must answer two questions.*?(?=\.|$)',  # Remove "You must answer two questions..." up to period or end
        r'You must answer three questions.*?(?=\.|$)',  # Remove "You must answer three questions..." up to period or end
        r'You must answer four questions.*?(?=\.|$)',  # Remove "You must answer four questions..." up to period or end
        r'You must answer five questions.*?(?=\.|$)',  # Remove "You must answer five questions..." up to period or end
        r'You must answer six questions.*?(?=\.|$)',  # Remove "You must answer six questions..." up to period or end
        r'You must answer seven questions.*?(?=\.|$)',  # Remove "You must answer seven questions..." up to period or end
        r'You must answer eight questions.*?(?=\.|$)',  # Remove "You must answer eight questions..." up to period or end
        r'You must answer nine questions.*?(?=\.|$)',  # Remove "You must answer nine questions..." up to period or end
        r'You must answer ten questions.*?(?=\.|$)',  # Remove "You must answer ten questions..." up to period or end
        r'You must answer eleven questions.*?(?=\.|$)',  # Remove "You must answer eleven questions..." up to period or end
        r'You must answer twelve questions.*?(?=\.|$)',  # Remove "You must answer twelve questions..." up to period or end
        r'You must answer thirteen questions.*?(?=\.|$)',  # Remove "You must answer thirteen questions..." up to period or end
        r'You must answer fourteen questions.*?(?=\.|$)',  # Remove "You must answer fourteen questions..." up to period or end
        r'You must answer fifteen questions.*?(?=\.|$)',  # Remove "You must answer fifteen questions..." up to period or end
        r'You must answer sixteen questions.*?(?=\.|$)',  # Remove "You must answer sixteen questions..." up to period or end
        r'You must answer seventeen questions.*?(?=\.|$)',  # Remove "You must answer seventeen questions..." up to period or end
        r'You must answer eighteen questions.*?(?=\.|$)',  # Remove "You must answer eighteen questions..." up to period or end
        r'You must answer nineteen questions.*?(?=\.|$)',  # Remove "You must answer nineteen questions..." up to period or end
        r'You must answer twenty questions.*?(?=\.|$)',  # Remove "You must answer twenty questions..." up to period or end
    ]
    question_artifact_regex = re.compile("|".join(question_artifact_patterns), re.IGNORECASE)

    while idx < text_len:
        match = start_pattern.search(text, idx)
        if not match:
            break
        marker = match.group(0)
        start_idx = match.end()

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
            idx = start_idx
            continue

        end_match = end_regex.search(text, start_idx)
        next_start = find_next_start(text, start_idx)
        if end_match and end_match.start() < next_start:
            content_end = end_match.start()
            next_idx = end_match.end()
        else:
            content_end = next_start
            next_idx = next_start

        content = text[start_idx:content_end].strip()

        # Heuristic fix for tiny fragments inside question markers
        if key == "question" and len(content) < 5:
            trailing_end = next_start
            trailing_text = text[content_end:trailing_end].strip()
            if trailing_text and not trailing_text.startswith("$"):
                content = (content + " " + trailing_text).strip()

        # 🔍 NEW: Capture question text immediately after $$END if it looks like a question
        if key == "question":
            lookahead_text = text[content_end:next_idx].strip()
            if lookahead_text and question_like.search(lookahead_text):
                content = (content + " " + lookahead_text).strip()

        if key == "question":
            # Detect placeholder like "(a)", "(α)", "a)", etc.
            placeholder_like = re.match(r'^\(?[a-zα-ω]\)?$', content.strip(), re.IGNORECASE)
            is_too_short = len(content.strip()) < 5 and not question_like.search(content)

            if placeholder_like or is_too_short:
                # Look ahead after $$END for real question text
                lookahead_text = text[content_end:next_idx].strip()
                if lookahead_text and question_like.search(lookahead_text):
                    content = lookahead_text
                else:
                    content = "UNKNOWN_QUESTION"
            else:
                # Normal case: maybe extend with nearby text
                lookahead_text = text[content_end:next_idx].strip()
                if lookahead_text and question_like.search(lookahead_text):
                    content = (content + " " + lookahead_text).strip()

        # Save extracted range
        mapped_ranges.append((start_idx, content_end))

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
        # elif key == "answer":
        #     current[key] = content
        #     result.append(current)
        #     current = {}

        elif key == "answer":
            # Detect mis-nested question inside the answer block
            question_match = re.search(r'(.+?\?\s*|\bName the\b.*?\.\s*)\$\$END', content, re.IGNORECASE | re.DOTALL)
            if question_match:
                # Split into question + real answer
                q_text = question_match.group(1).replace("$$END", "").strip()
                a_text = content[question_match.end():].strip()
                current["question"] = q_text
                current["answer"] = a_text
            else:
                current["answer"] = content
            result.append(current)
            current = {}

        idx = next_idx

    if current:
        result.append(current)

    # -----------------------------------------------------
    # 🔍 Recovery step: find answers without questions
    # -----------------------------------------------------
    unmapped_regions = []
    last_end = 0
    for start, end in sorted(mapped_ranges):
        if last_end < start:
            unmapped_regions.append(text[last_end:start].strip())
        last_end = end
    if last_end < len(text):
        unmapped_regions.append(text[last_end:].strip())

    recovered_questions = []
    for region in unmapped_regions:
        if question_like.search(region):
            recovered_questions.append(region)

    # Attach recovered questions to dangling answers
    final_result = []
    q_idx = 0
    for item in result:
        if "answer" in item and "question" not in item:
            if q_idx < len(recovered_questions):
                item["question"] = recovered_questions[q_idx]
                q_idx += 1
            else:
                item["question"] = "UNKNOWN_QUESTION"

        elif "question" in item and item["question"] == "UNKNOWN_QUESTION":
            if q_idx < len(recovered_questions):
                item["question"] = recovered_questions[q_idx]
                q_idx += 1

        # 🔹 Apply cleanup here
        # Clean up question artifacts
        cleaned_question = clean_text(item.get("question", ""))
        cleaned_question = question_artifact_regex.sub("", cleaned_question)
        cleaned_question = re.sub(r"\s+", " ", cleaned_question).strip()

        cleaned_answer = clean_text(item.get("answer", ""))

        item["question"] = cleaned_question
        item["answer"] = cleaned_answer

        final_result.append(item)

    return final_result

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
