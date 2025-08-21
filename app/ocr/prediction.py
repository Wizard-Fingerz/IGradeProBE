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

from app.ocr.prediction2 import QuestionTypePredictor

# Load embedding model once
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

warnings.simplefilter("ignore")

# Load spaCy's medium-sized English language model
nlp = spacy.load("en_core_web_md")

# Load semantic model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

from transformers import pipeline

# Load a small open-source LLM locally (e.g., Mistral or LLaMA-2-7B if you have GPU)
llm = pipeline("text-generation", model="google/flan-t5-large")



ROMAN_LABEL_RE = re.compile(r'\b([IVXLCDM]+)\s*:\s*([^\n\r]+?)(?=(?:\b[IVXLCDM]+\s*:)|$)', re.IGNORECASE)


class PredictionService:
    def __init__(self):
        self.model = self.load_model()
        self.qtype_predictor = QuestionTypePredictor()


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

    def _first_percent_number(self, text: str):
        m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*%?', text)
        return float(m.group(1)) if m else None

    def _clean_text(self, s: str):
        return re.sub(r'[\s\.,;:()\[\]-]+', ' ', s).strip().lower()

    def _extract_label_map(self, text: str, expected_labels=None):
        """
        Extract {label -> value_text} from 'III: ... IV: ...' style text.
        If expected_labels is given, only keep those labels.
        """
        mapping = {}
        if not isinstance(text, str):
            return mapping

        for lab, val in ROMAN_LABEL_RE.findall(text):
            lab_norm = lab.upper()
            if expected_labels and lab_norm not in expected_labels:
                continue
            mapping[lab_norm] = val.strip()

        # If nothing matched but we know which labels we expect, try a loose fallback:
        # split by spaces and try to assign first values following each label token.
        if not mapping and expected_labels:
            tokens = text.split()
            labs = [t.rstrip(':').upper() for t in tokens if t.rstrip(':').upper() in expected_labels]
            for i, tok in enumerate(tokens):
                key = tok.rstrip(':').upper()
                if key in expected_labels and key not in mapping:
                    # grab the slice after the label up to next known label
                    j = i + 1
                    chunk = []
                    while j < len(tokens) and tokens[j].rstrip(':').upper() not in expected_labels:
                        chunk.append(tokens[j])
                        j += 1
                    if chunk:
                        mapping[key] = ' '.join(chunk).strip()
        return mapping

    # def _label_map_similarity(self, student_text: str, examiner_text: str, comprehension_text: str = None,
    #                         percent_tolerance: float = 0.0,  # no tolerance
    #                         text_fuzzy_thresh: int = 100,    # exact match only
    #                         embed_thresh: float = 1.0) -> float:
    #     """
    #     Strict Label → Value comparison:
    #     - Only exact matches count (fuzzy ratio = 100)
    #     - Percent tolerance = 0
    #     """
    #     exam_map = self._extract_label_map(examiner_text)
    #     if not exam_map:
    #         return 0.0
    #     labels = list(exam_map.keys())

    #     stu_map = self._extract_label_map(student_text, expected_labels=set(labels))

    #     correct = 0
    #     for lab in labels:
    #         expected_val = self._clean_text(exam_map[lab])
    #         student_val  = stu_map.get(lab, None)
    #         if student_val is None:
    #             continue

    #         student_val = self._clean_text(student_val)

    #         # Only exact match counts
    #         if expected_val == student_val:
    #             correct += 1

    #     return correct / max(len(labels), 1)


    # def _label_map_similarity_strict_order(self, student_text: str, examiner_text: str) -> float:
    #     """
    #     Label-to-Value grading with strict label order.
    #     Partial credit allowed for each correct label in the correct sequence.
    #     Works for numbers, percentages, or text.
    #     """
    #     # Extract ordered label->value lists
    #     exam_matches = list(ROMAN_LABEL_RE.findall(examiner_text))
    #     student_matches = list(ROMAN_LABEL_RE.findall(student_text))

    #     total_labels = len(exam_matches)
    #     if total_labels == 0:
    #         return 0.0

    #     correct_labels = 0

    #     # Compare in order
    #     for i, (lab, val) in enumerate(exam_matches):
    #         if i < len(student_matches):
    #             stu_lab, stu_val = student_matches[i]

    #             # Check label matches (optional)
    #             if lab.upper() != stu_lab.upper():
    #                 continue

    #             # Numeric comparison
    #             try:
    #                 if float(val) == float(stu_val):
    #                     correct_labels += 1
    #                     continue
    #             except ValueError:
    #                 pass

    #             # Text comparison fallback
    #             if val.strip().lower() == stu_val.strip().lower():
    #                 correct_labels += 1

    #     return correct_labels / total_labels



    # def _strict_label_map_similarity(self, student_text: str, examiner_text: str) -> float:
    #     """
    #     Strict comparison for Label_to_Value:
    #     - Only exact matches count
    #     - Fractional score: correct_labels / total_labels
    #     """
    #     exam_map = self._extract_label_map(examiner_text)
    #     if not exam_map:
    #         return 0.0
        
    #     stu_map = self._extract_label_map(student_text, expected_labels=set(exam_map.keys()))

    #     correct = 0
    #     for label, expected_val in exam_map.items():
    #         expected_val_clean = expected_val.strip().lower()
    #         student_val = stu_map.get(label, None)
    #         if student_val is None:
    #             continue
    #         student_val_clean = student_val.strip().lower()
    #         if expected_val_clean == student_val_clean:
    #             correct += 1

    #     return correct / max(len(exam_map), 1)

    def _strict_label_map_similarity(self, student_text: str, examiner_text: str) -> float:
        """
        Strict comparison for Label_to_Value:
        - Accepts 99% similarity (fuzzy match) instead of only exact matches
        - Fractional score: correct_labels / total_labels
        """

        exam_map = self._extract_label_map(examiner_text)
        if not exam_map:
            return 0.0
        
        stu_map = self._extract_label_map(student_text, expected_labels=set(exam_map.keys()))

        correct = 0
        for label, expected_val in exam_map.items():
            expected_val_clean = expected_val.strip().lower()
            student_val = stu_map.get(label, None)
            if student_val is None:
                continue
            student_val_clean = student_val.strip().lower()
            # Accept 99% similarity instead of only exact match
            if fuzz.ratio(expected_val_clean, student_val_clean) >= 90:
                correct += 1

        return correct / max(len(exam_map), 1)



    # def list_similarity(self, student_answer: str, examiner_answer: str, comprehension_text: str) -> float:
    #     """
    #     Compare list-type answers.
    #     Returns similarity score (0–1) based on how many correct items student provides.
    #     """

    #     # --- Normalize ---
    #     examiner_items = [x.strip().lower() for x in examiner_answer.split(",")]
    #     comprehension_words = set(w.strip().lower() for w in comprehension_text.replace("\n", " ").split())

    #     # Student answer -> break into lines / commas / semicolons
    #     raw_items = student_answer.replace("\n", ",").replace(";", ",").split(",")
    #     student_items = [x.strip().lower() for x in raw_items if x.strip()]

    #     # --- Matching ---
    #     correct = 0
    #     matched = set()

    #     for s_item in student_items:
    #         # 1. Direct match with examiner items
    #         for e_item in examiner_items:
    #             if fuzz.token_set_ratio(s_item, e_item) >= 90 and e_item not in matched:
    #                 correct += 1
    #                 matched.add(e_item)
    #                 break
    #         else:
    #             # 2. If not in examiner's list, check if valid from comprehension
    #             for word in comprehension_words:
    #                 if fuzz.token_set_ratio(s_item, word) >= 90:
    #                     correct += 1
    #                     break

    #     # --- Score ---
    #     required = len(examiner_items)
    #     score = min(correct / required, 1.0)  # cap at 1.0

    #     return score

    def list_similarity(
            self, 
            student_answer: str, 
            examiner_answer: str, 
            comprehension_text: str, 
            question_text: str
        ) -> float:
        """
        Compare list-type answers using sentence embeddings.
        Returns similarity score (0–1) based on how many correct items student provides,
        normalized by the number of expected points.
        """

        # --- Normalize examiner’s list ---
        examiner_items = [x.strip() for x in examiner_answer.replace("\n", ",").split(",") if x.strip()]

        # --- Comprehension as points ---
        comp_items = [x.strip() for x in comprehension_text.replace("\n", ",").split(",") if x.strip()]

        # --- Normalize student’s list ---
        student_items = [x.strip() for x in student_answer.replace("\n", ",").replace(";", ",").split(",") if x.strip()]

        # --- Encode everything ---
        examiner_embs = sentence_model.encode(examiner_items, convert_to_tensor=True) if examiner_items else []
        comp_embs = sentence_model.encode(comp_items, convert_to_tensor=True) if comp_items else []
        student_embs = sentence_model.encode(student_items, convert_to_tensor=True) if student_items else []

        correct = 0
        matched = set()

        # --- Matching logic ---
        for i, s_emb in enumerate(student_embs):
            best_sim = 0
            best_match = None

            # 1. Check similarity with examiner’s items
            if len(examiner_embs) > 0:
                sims_exam = util.cos_sim(s_emb, examiner_embs)[0]
                best_exam_idx = int(sims_exam.argmax())
                best_exam_sim = float(sims_exam[best_exam_idx])
                if best_exam_sim > best_sim:
                    best_sim = best_exam_sim
                    best_match = ("examiner", best_exam_idx)

            # 2. If not high enough, check comprehension
            if len(comp_embs) > 0:
                sims_comp = util.cos_sim(s_emb, comp_embs)[0]
                best_comp_idx = int(sims_comp.argmax())
                best_comp_sim = float(sims_comp[best_comp_idx])
                if best_comp_sim > best_sim:
                    best_sim = best_comp_sim
                    best_match = ("comprehension", best_comp_idx)

            # Threshold for considering it correct
            if best_sim >= 0.7:  # tune threshold (0.65–0.75 works well)
                if best_match not in matched:  # avoid duplicate matching
                    correct += 1
                    matched.add(best_match)

        # --- Denominator: expected points ---
        expected_points = self.extract_expected_points(question_text)
        if not expected_points:
            expected_points = len(examiner_items) if examiner_items else 1

        # --- Compute score ---
        score = min(correct / expected_points, 1.0)

        return score

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


    def extract_expected_points(self, question_text: str) -> int:
        """
        Extracts the number of points required from the question.
        Example: "Explain five importance of democracy" -> returns 5
        """
        match = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", question_text, re.IGNORECASE)
        if not match:
            return 0
        
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        value = match.group(0).lower()
        if value.isdigit():
            return int(value)
        return word_to_num.get(value, 0)

    def preprocess_list_and_explain_student_answer(self, student_answer):
        """
        Groups multi-line explanations into single text blocks
        based on the format 'Point: explanation'.
        """
        grouped = []
        current = ""

        for line in student_answer.split("\n"):
            line = line.strip()
            if not line:
                continue

            # If line looks like a new point (has ':' or numbering)
            if re.match(r"^[a-zA-Z].*?:", line) or re.match(r"^\d+[\).]", line):
                if current:
                    grouped.append(current.strip())
                current = line
            else:
                # continuation of the previous point
                current += " " + line

        if current:
            grouped.append(current.strip())

        return grouped

    def list_and_explain_similarity(self, question_text, student_answer, examiner_answer, comprehension, question_score):
        """
        Evaluate a student's List & Explain answer with semantic similarity.
        Returns a normalized score (0–1) and per-point details.
        """

        print("Student answer received:", student_answer)

        # --- Step 1: Extract expected number of points ---
        expected_points = self.extract_expected_points(question_text)

        # --- Step 2: Split examiner points ---
        examiner_points = [p.strip(" .-") for p in examiner_answer.split("\n") if p.strip()]
        if not examiner_points:
            return {"score": 0.0, "details": []}

        if expected_points == 0:
            expected_points = len(examiner_points)

        # --- Step 3: Preprocess student answer into smaller sentences ---
        import re
        stu_sentences = re.split(r'[.;)\n•\d]+', student_answer)
        stu_sentences = [s.strip() for s in stu_sentences if s.strip()]
        if not stu_sentences:
            return {"score": 0.0, "details": []}

        # --- Step 4: Encode comprehension (split into sentences for robustness) ---
        comp_sentences = re.split(r'[.;)\n]+', comprehension)
        comp_sentences = [s.strip() for s in comp_sentences if s.strip()]
        comp_embeddings = [sentence_model.encode(s, convert_to_tensor=True) for s in comp_sentences]

        # --- Step 5: Set thresholds ---
        LIST_THRESHOLD = 0.45
        EXPLAIN_THRESHOLD = 0.35

        point_scores = []
        details = []

        # --- Step 6: Compare each examiner point to student sentences ---
        # --- Step 6: Compare each examiner point to student sentences ---
        # Original: for idx, point in enumerate(examiner_points[:expected_points]):
        # New: iterate over all student sentences and check against all examiner points

        matched_examiner = set()  # keep track of already matched examiner points

        for s in stu_sentences:
            s_emb = sentence_model.encode(s, convert_to_tensor=True)

            best_score = 0.0
            best_idx = -1
            best_sim_point = 0.0
            best_sim_comp = 0.0

            for idx, point in enumerate(examiner_points):
                if idx in matched_examiner:
                    continue  # skip already matched examiner point

                point_emb = sentence_model.encode(point, convert_to_tensor=True)
                sim_with_point = util.cos_sim(point_emb, s_emb).item()
                sim_with_comp = max([util.cos_sim(ce, s_emb).item() for ce in comp_embeddings])

                score = 0.0
                if sim_with_point > LIST_THRESHOLD:
                    score = 0.33
                    if sim_with_comp > EXPLAIN_THRESHOLD:
                        score += 0.67

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_sim_point = sim_with_point
                    best_sim_comp = sim_with_comp

            if best_idx != -1:
                matched_examiner.add(best_idx)
                point_scores.append(best_score)
                details.append({
                    "examiner_point": examiner_points[best_idx],
                    "student_best_match": s,
                    "similarity_with_point": round(best_sim_point, 3),
                    "similarity_with_comprehension": round(best_sim_comp, 3),
                    "score_for_point": round(best_score, 2)
                })

        # --- Step 7: Pad unmatched examiner points with 0 ---
        for idx, point in enumerate(examiner_points):
            if idx not in matched_examiner:
                point_scores.append(0.0)
                details.append({
                    "examiner_point": point,
                    "student_best_match": "",
                    "similarity_with_point": 0.0,
                    "similarity_with_comprehension": 0.0,
                    "score_for_point": 0.0
                })


        # --- Step 8: Average score across points ---
        avg_similarity = sum(point_scores) / expected_points if expected_points else 0.0

        avg_similarity = min(avg_similarity, 1)

        # Debug print
        for d in details:
            print(f"Examiner: {d['examiner_point']}\n"
                f"Student: {d['student_best_match']}\n"
                f"Sim(Point): {d['similarity_with_point']}, "
                f"Sim(Comp): {d['similarity_with_comprehension']}, "
                f"Score: {d['score_for_point']}\n---")

        return avg_similarity




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
       
       
        # Step 0: Predict question type
        qtype = self.qtype_predictor.predict(question)

        print("[INFO] Question Type of", question, "is", qtype)


        # Step 1: Keyword overlap
        overlap_count, precision, recall, f1 = self.keyword_overlap(student_answer, examiner_answer)
        
        # Step 2: Decide similarity function
        if qtype == "Label_to_Value":
            strict_semantic_similarity = self._strict_label_map_similarity(student_answer, examiner_answer)
            
            print("Semantic analysis", strict_semantic_similarity)
            features = np.array([[strict_semantic_similarity, question_score]])
            strict_model_score = float(self.model.predict(features)[0])
            return strict_model_score
            #  we should exempt label with list
        # elif qtype in ["List / Enumeration"]:
        #     semantic_similarity = self.list_similarity(student_answer, examiner_answer, comprehension)
        elif qtype in ["List / Enumeration"]:

            
            semantic_similarity = self.list_similarity(student_answer, examiner_answer, comprehension, question)
        
        elif qtype in ["List and Explain"]:
            list_and_explain_semantic_similarity = self.list_and_explain_similarity(question, student_answer, examiner_answer, comprehension, question_score)
            print("list_and_explain_semantic_similarity", list_and_explain_semantic_similarity)

            list_model_score = float(list_and_explain_semantic_similarity * question_score)

            print('Prediction After similarity',list_model_score)
            print('Question Score',question_score)
            return list_model_score
        
       
        else:  # Text / Comprehension / Definition / Others
            weights = {'examiner': 0.1, 'comprehension': 0.9}
            semantic_similarity = self.calculate_combined_similarity(student_answer, examiner_answer, comprehension, weights)

            print("semantic_similarity", semantic_similarity)


        adjusted_similarity = (semantic_similarity * 0.8) + (f1 * 0.2)


        # Step 3: Build feature vector (ONLY 2 features)
        features = np.array([[adjusted_similarity, question_score]])

        

        # Step 4: Model prediction
        model_score = float(self.model.predict(features)[0])

        print("model_score",model_score)

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



