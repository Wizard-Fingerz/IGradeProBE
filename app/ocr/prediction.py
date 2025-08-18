import joblib
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sentence_transformers import SentenceTransformer, util
# from fuzzywuzzy import fuzz, process
from rapidfuzz import fuzz, process
import re

# Load embedding model once
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

warnings.simplefilter("ignore")

# Load spaCy's medium-sized English language model
nlp = spacy.load("en_core_web_md")

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

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

    # def list_similarity(self, student_items, examiner_items, comprehension_items=None):
    #     """
    #     Compute similarity score for list-type questions.
    #     Uses examiner answers as the gold standard but allows matches
    #     from comprehension as valid alternatives.
    #     Returns only a similarity score (0-1).
    #     """
    #     if comprehension_items is None:
    #         comprehension_items = []

    #     # Normalize sets
    #     examiner_set = set(item.lower().strip() for item in examiner_items)
    #     comprehension_set = set(item.lower().strip() for item in comprehension_items)
    #     student_set = set(item.lower().strip() for item in student_items)

    #     # Union of possible correct answers
    #     valid_answers = examiner_set.union(comprehension_set)

    #     total_required = len(examiner_set)
    #     if total_required == 0:
    #         return 0.0

    #     score = 0
    #     matched = set()

    #     for student_item in student_set:
    #         # Direct or fuzzy match against valid answers
    #         for valid_item in valid_answers:
    #             if valid_item in matched:
    #                 continue
    #             if student_item == valid_item or fuzz.ratio(student_item, valid_item) > 80:
    #                 score += 1
    #                 matched.add(valid_item)
    #                 break

    #     return round(score / total_required, 2)


    def list_similarity(self, student_answer: str, examiner_answer: str, comprehension_text: str) -> float:
        """
        Compare list-type answers.
        Returns similarity score (0–1) based on how many correct items student provides.
        """

        # --- Normalize ---
        examiner_items = [x.strip().lower() for x in examiner_answer.split(",")]
        comprehension_words = set(w.strip().lower() for w in comprehension_text.replace("\n", " ").split())

        # Student answer -> break into lines / commas / semicolons
        raw_items = student_answer.replace("\n", ",").replace(";", ",").split(",")
        student_items = [x.strip().lower() for x in raw_items if x.strip()]

        # --- Matching ---
        correct = 0
        matched = set()

        for s_item in student_items:
            # 1. Direct match with examiner items
            for e_item in examiner_items:
                if fuzz.token_set_ratio(s_item, e_item) >= 90 and e_item not in matched:
                    correct += 1
                    matched.add(e_item)
                    break
            else:
                # 2. If not in examiner's list, check if valid from comprehension
                for word in comprehension_words:
                    if fuzz.token_set_ratio(s_item, word) >= 90:
                        correct += 1
                        break

        # --- Score ---
        required = len(examiner_items)
        score = min(correct / required, 1.0)  # cap at 1.0

        return score

   
   
    # def list_similarity(self, student_items, examiner_items, comprehension_items=None, threshold=0.8):
    #     # Ensure inputs are lists
    #     if isinstance(examiner_items, str):
    #         examiner_items = [examiner_items]
    #     if isinstance(student_items, str):
    #         student_items = [student_items]
    #     if comprehension_items and isinstance(comprehension_items, str):
    #         comprehension_items = [comprehension_items]

    #     # Merge examiner + comprehension (if available)
    #     reference_items = examiner_items[:]
    #     if comprehension_items:
    #         reference_items.extend(comprehension_items)

    #     # Encode
    #     ref_emb = model.encode(reference_items, convert_to_tensor=True)
    #     stu_emb = model.encode(student_items, convert_to_tensor=True)

    #     # Compute similarity scores
    #     scores = []
    #     for i in range(len(student_items)):
    #         sims = util.cos_sim(stu_emb[i].unsqueeze(0), ref_emb)  # (1, N)
    #         best_score = sims.max().item()  # highest similarity for this student item
    #         scores.append(best_score)

        
    #     # Aggregate into a single score (mean)
    #     aggregate_score = sum(scores) / len(scores) if scores else 0.0

    #     return aggregate_score


    def is_list_question(self, question_text: str) -> bool:
        """
        Detects if a question expects a list answer (e.g., 'Name three...', 'List four...').
        """
        patterns = [
            r"\bname\s+\d+", 
            r"\blist\s+\d+", 
            r"\bgive\s+\d+", 
            r"\bmention\s+\d+",
            r"\bstate\s+\d+"
        ]
        q_lower = question_text.lower()
        return any(re.search(p, q_lower) for p in patterns)


    def evaluate_answer(self, question, student_answer, examiner_answer, comprehension_items=None):
        """
        Automatically decides whether to use list similarity or text similarity
        based on the question type.
        """
        if self.is_list_question(question):
            return self.list_similarity(student_answer, examiner_answer, comprehension_items)
        else:
            # fallback for normal text answers (fuzzy match)
            return round(fuzz.ratio(" ".join(student_answer), " ".join(examiner_answer)) / 100, 2)

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
            weights = {'examiner': 0.1, 'comprehension': 0.9}
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

