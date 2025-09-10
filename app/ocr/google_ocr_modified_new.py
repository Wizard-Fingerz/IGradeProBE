import logging
from difflib import get_close_matches
import io
from google.cloud import vision
from google.oauth2 import service_account

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

# ------------------------------------------------------------------------
# Approaches for Matching Questions to Answers WITHOUT Regex or Delimiter-based Splitting
# ------------------------------------------------------------------------

"""
Below are several additional approaches for matching questions to answers without using regex
and without relying on explicit delimiters like "Q:", "A:", "Question", or "Answer".

1. **Line-by-Line Heuristics**
   - Iterate through each line of the text.
   - Treat lines ending with a question mark as questions.
   - The following non-empty lines, until the next question, are considered the answer.
   - Example:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        qa_pairs = []
        i = 0
        while i < len(lines):
            if lines[i].endswith('?'):
                question = lines[i]
                answer = ""
                i += 1
                while i < len(lines) and not lines[i].endswith('?'):
                    answer += lines[i] + " "
                    i += 1
                qa_pairs.append({'question': question, 'answer': answer.strip()})
            else:
                i += 1

2. **Numbered List or Enumeration Parsing**
   - Detect lines that start with a number and a period or parenthesis (e.g., "1.", "2)", "3.").
   - Treat these as the start of a new question.
   - The text following each number is the question, and the subsequent lines (until the next number) are the answer.
   - Example:
        lines = text.split('\n')
        qa_pairs = []
        current_q = None
        current_a = []
        for line in lines:
            s = line.strip()
            if s and s[0].isdigit() and (len(s) > 1 and s[1] in ['.', ')']):
                if current_q:
                    qa_pairs.append({'question': current_q, 'answer': ' '.join(current_a).strip()})
                current_q = s
                current_a = []
            else:
                if current_q:
                    current_a.append(s)
        if current_q:
            qa_pairs.append({'question': current_q, 'answer': ' '.join(current_a).strip()})

3. **Whitespace/Block-Based Grouping**
   - Split the text into blocks using double newlines or multiple blank lines.
   - Assume each block contains either a question or an answer.
   - Alternate blocks as question/answer pairs.
   - Example:
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        qa_pairs = []
        for i in range(0, len(blocks)-1, 2):
            qa_pairs.append({'question': blocks[i], 'answer': blocks[i+1]})

4. **Keyword and Structure Heuristics**
   - Use common question words (Who, What, When, Where, Why, How, etc.) at the start of a line to identify questions.
   - Use the next contiguous lines as the answer.
   - Example:
        question_words = ('Who', 'What', 'When', 'Where', 'Why', 'How')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        qa_pairs = []
        i = 0
        while i < len(lines):
            if any(lines[i].startswith(qw) for qw in question_words) and lines[i].endswith('?'):
                question = lines[i]
                answer = ""
                i += 1
                while i < len(lines) and not (any(lines[i].startswith(qw) for qw in question_words) and lines[i].endswith('?')):
                    answer += lines[i] + " "
                    i += 1
                qa_pairs.append({'question': question, 'answer': answer.strip()})
            else:
                i += 1

5. **Punctuation and Sentence Structure Analysis**
   - Use sentence segmentation (e.g., with NLTK or spaCy) to split text into sentences.
   - Classify sentences as questions if they end with a question mark.
   - Group each question with the following sentences as its answer, until the next question is found.

6. **Layout or Indentation Cues**
   - If the OCR output preserves indentation or bullet points, use these to distinguish between questions and answers.
   - For example, unindented lines may be questions, and indented or bulleted lines may be answers.

7. **Font or Style Information (if available)**
   - If the OCR output includes font size, bold, or italic information, use these cues to distinguish questions (e.g., bold or larger font) from answers.

8. **Proximity and Length Heuristics**
   - Treat short lines ending with a question mark as questions.
   - Longer lines or paragraphs following them are likely answers.

9. **Table or Column Structure**
   - If the OCR output preserves table or column structure, treat each row as a question/answer pair, or use the first column as questions and the second as answers.

10. **Advanced Machine Learning/NLP Approaches for Robust Q/A Extraction (Handling Implicit Questions)**

    - Recognize that exam-style questions often lack explicit punctuation (e.g., question marks) and may begin with directive verbs ("state", "mention", "list", "explain", etc.), making rule-based extraction unreliable.
    - Employ a multi-stage NLP pipeline that leverages both classical and deep learning techniques for robust question/answer segmentation:

    **a. Preprocessing and Feature Engineering**
        - Tokenize the OCR text into sentences or lines using advanced sentence boundary detection (e.g., spaCy, NLTK Punkt, or transformer-based segmenters).
        - For each line/sentence, extract a rich set of features:
            - Lexical: presence of question words (including expanded lists: "state", "mention", "list", "explain", "describe", "define", "give", "outline", "identify", "cite", "enumerate", "write", "name", "discuss", "show", "prove", "why", "how", "what", "when", "where", "who", "which", etc.)
            - Syntactic: part-of-speech tags, dependency parse (e.g., does the sentence start with a verb? Is it imperative or interrogative?)
            - Semantic: sentence embeddings (e.g., using BERT, RoBERTa, or Sentence Transformers)
            - Structural: position in the document, length, indentation, bullet points, font/style cues (if available)
            - Contextual: features from neighboring lines (e.g., does the previous line look like a question?)

    **b. Sequence Labeling/Classification**
        - Use a sequence labeling model (e.g., BiLSTM-CRF, transformer-based token/sentence classifier) to assign each line/sentence a label: "QUESTION", "ANSWER", or "OTHER".
        - Optionally, use a hierarchical model that considers both sentence-level and block-level context.
        - For best results, fine-tune a transformer model (e.g., BERT, DeBERTa, or Longformer for long documents) on a labeled dataset of exam-style Q&A pairs, including examples without explicit question marks.

    **c. Postprocessing and Q/A Pair Construction**
        - Iterate through the labeled sequence:
            - When a "QUESTION" label is detected, start a new Q/A pair.
            - Aggregate subsequent "ANSWER" lines until the next "QUESTION" or "OTHER" is found.
            - Optionally, merge short "OTHER" lines with the preceding answer if they appear to be part of the answer context.
        - Handle edge cases, such as multi-part questions, sub-questions, or answers spanning multiple paragraphs.

    **d. Optional Enhancements**
        - Use cross-sentence coreference resolution to improve answer boundary detection.
        - Integrate document layout analysis (e.g., using OCR bounding boxes) to leverage spatial relationships between questions and answers.
        - Apply unsupervised clustering (e.g., using sentence embeddings) to group similar questions or answers for further validation.

    **e. Example Workflow (Pseudocode)**
        ```
        sentences = advanced_sentence_split(ocr_text)
        features = [extract_features(sent, context=sentences, idx=i) for i, sent in enumerate(sentences)]
        labels = sequence_labeling_model.predict(features)
        qa_pairs = []
        i = 0
        while i < len(sentences):
            if labels[i] == "QUESTION":
                question = sentences[i]
                answer = ""
                i += 1
                while i < len(sentences) and labels[i] == "ANSWER":
                    answer += sentences[i] + " "
                    i += 1
                qa_pairs.append({'question': question, 'answer': answer.strip()})
            else:
                i += 1
        ```

    - This approach enables accurate extraction of Q/A pairs even in challenging, unstructured, or punctuation-deficient exam documents, and can be further improved with domain-specific training data and layout-aware models.




# ------------------------------------------------------------------------
# Example: Simple Non-Regex, Non-Delimiter Question/Answer Extraction
# ------------------------------------------------------------------------
"""
