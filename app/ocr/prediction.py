import joblib
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz, process


# Load embedding model once
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

warnings.simplefilter("ignore")

# Load spaCy's medium-sized English language model
nlp = spacy.load("en_core_web_md")

from transformers import pipeline

# Load a small open-source LLM locally (e.g., Mistral or LLaMA-2-7B if you have GPU)
llm = pipeline("text-generation", model="google/flan-t5-large")


class PredictionService:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model_path = './new_dump/dt_model_new.joblib'
        try:
            model = joblib.load(model_path)
            print('✅ Model loaded successfully')
            return model
        except Exception as e:
            print(f"❌ Error loading the model: {e}")
            return None

    def preprocess_text(self, text):
        """Lowercase, remove punctuation, tokenize, stopword removal, lemmatization"""
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def keyword_overlap(self, student_answer, examiner_answer):
        """Calculate precision, recall, F1 for overlap"""
        student_tokens = set(self.preprocess_text(student_answer).split())
        examiner_tokens = set(self.preprocess_text(examiner_answer).split())
        overlap = student_tokens.intersection(examiner_tokens)

        overlap_count = len(overlap)
        precision = overlap_count / max(len(student_tokens), 1)
        recall = overlap_count / max(len(examiner_tokens), 1)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return overlap_count, precision, recall, f1


    def list_similarity(self, student_answer, examiner_answer, comprehension):
        """
        Handle list-style questions (e.g., 'List 3 causes of...').
        Returns a similarity score [0,1].
        """
        # Split examiner into list points
        points = [p.strip(" -•") for p in examiner_answer.split("\n") if p.strip()]
        if len(points) == 1:
            points = [p.strip() for p in examiner_answer.split(" - ") if p.strip()]

        # If not a list, just fall back to normal similarity
        if len(points) <= 1:
            return self.calculate_combined_similarity(student_answer, examiner_answer, comprehension)

        student_items = [s.strip() for s in student_answer.split("\n") if s.strip()]
        if len(student_items) == 1:  
            # fallback: space-separated
            student_items = [s.strip() for s in student_answer.split() if s.strip()]

        matches = 0
        matched_exam_points = set()

        for s_item in student_items:
            # fuzzy match student item against examiner points
            best_match = process.extractOne(s_item, points, scorer=fuzz.token_sort_ratio)
            if best_match and best_match[1] >= 70:  # threshold
                exam_item = best_match[0]
                if exam_item not in matched_exam_points:
                    matched_exam_points.add(exam_item)
                    matches += 1

        coverage_ratio = matches / len(points)

        # Blend fuzzy match coverage with semantic similarity
        semantic_sim = self.calculate_combined_similarity(student_answer, examiner_answer, comprehension)
        list_similarity_score = (0.6 * coverage_ratio) + (0.4 * semantic_sim)

        return min(max(list_similarity_score, 0.0), 1.0)





    def refine_with_llm(self, question, student_answer, examiner_answer, comprehension, raw_score, max_score):
        """Optional: LLM reasoning refinement"""
        prompt = f"""
        You are grading a student answer.
        Comprehension Passage: {comprehension}
        Question: {question}
        Examiner Answer: {examiner_answer}
        Student Answer: {student_answer}
        Raw Score (from embeddings + keyword overlap): {raw_score:.2f} / {max_score}

        Task: Adjust the score if necessary, considering:
        - relevance to the question,
        - semantic similarity to the examiner answer,
        - partial credit for partially correct or incomplete answers.

        Return ONLY a number between 0 and {max_score}.
        """
        response = llm(prompt, max_new_tokens=50, do_sample=False)
        try:
            refined_score = float([t for t in response[0]['generated_text'].split() if t.replace('.', '', 1).isdigit()][0])
            return min(max(refined_score, 0), max_score)  # clamp to [0, max_score]
        except:
            return raw_score  # fallback if parsing fails

    def calculate_combined_similarity(self, student_answer, examiner_answer, comprehension, weights=None):
        if not student_answer or not examiner_answer or not comprehension:
            return 0.0

        # Encode into embeddings
        emb_student = embedder.encode(student_answer, convert_to_tensor=True)
        emb_examiner = embedder.encode(examiner_answer, convert_to_tensor=True)
        emb_comprehension = embedder.encode(comprehension, convert_to_tensor=True)

        # Compute cosine similarities
        similarity_examiner = util.cos_sim(emb_student, emb_examiner).item()
        similarity_comprehension = util.cos_sim(emb_student, emb_comprehension).item()

        # Dynamic weights logic (same as your current code)
        if weights is None:
            base_examiner = 0.2
            base_comprehension = 0.8
            diff = similarity_examiner - similarity_comprehension

            if diff > 0:
                shift = min(diff * 0.5, 0.5)
                weights = {
                    'examiner': min(base_examiner + shift, 0.7),
                    'comprehension': max(base_comprehension - shift, 0.3)
                }
            else:
                shift = min(abs(diff) * 0.5, 0.5)
                weights = {
                    'examiner': max(base_examiner - shift, 0.3),
                    'comprehension': min(base_comprehension + shift, 0.7)
                }

        # Weighted similarity
        combined_similarity = (weights['examiner'] * similarity_examiner) + \
                              (weights['comprehension'] * similarity_comprehension)

        print(f"Examiner sim: {similarity_examiner:.3f}, Comprehension sim: {similarity_comprehension:.3f}, "
              f"Weights: {weights}, Combined: {combined_similarity:.3f}")

        return combined_similarity



    def predict(self, question_id, comprehension, question, examiner_answer, student_answer, question_score):
        """Hybrid scoring system with 2-feature input"""
        # Step 1: Keyword overlap
        overlap_count, precision, recall, f1 = self.keyword_overlap(student_answer, examiner_answer)

        # Step 2: Decide similarity function
        if "\n" in examiner_answer or " - " in examiner_answer:
            semantic_similarity = self.list_similarity(student_answer, examiner_answer, comprehension)
        else:
            weights = {'examiner': 0.3, 'comprehension': 0.7}
            semantic_similarity = self.calculate_combined_similarity(
                student_answer, examiner_answer, comprehension, weights
            )

        adjusted_similarity = (semantic_similarity * 0.8) + (f1 * 0.2)


        # Step 3: Build feature vector (ONLY 2 features)
        features = np.array([[adjusted_similarity, question_score]])

        # Step 4: Model prediction
        model_score = float(self.model.predict(features)[0])

        # Step 5: Rule-based override (full marks if coverage)
        if overlap_count >= question_score:
            return question_score  

        # Step 6: Clip model score
        model_score = min(model_score, question_score)

        # Step 7: Optional LLM refinement
        llm_score = self.refine_with_llm(
            question, student_answer, examiner_answer, comprehension,
            model_score, question_score
        )


        # Step 8: Decide final score
        # Rule: trust model if it's already high quality (close to max), 
        # otherwise let LLM improve it if valid
        if abs(model_score - question_score) <= 0.5:
            return model_score  # model already confident
        else:
            # Choose whichever score is closer to ground-truth constraints
            # e.g., within [0, question_score] and consistent with overlap
            if llm_score is not None:
                return min(max(llm_score, 0), question_score)
            return model_score

