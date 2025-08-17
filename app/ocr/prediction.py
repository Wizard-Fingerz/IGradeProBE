# import joblib
# import numpy as np
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import warnings

# warnings.simplefilter("ignore")
# import joblib
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.metrics.pairwise import cosine_similarity

# # Load spaCy's medium-sized English language model
# nlp = spacy.load("en_core_web_md")

# class PredictionService:
#     def __init__(self):
#         self.model = self.load_model()

    
#     def load_model(self):
#         model_path = './new_dump/dt_model_new.joblib'

#         try:
#             model = joblib.load(model_path)
#             print('I was succesful')
#             return model
#         except Exception as e:
#             print(f"Error loading the model: {e}")
#             return None


#     def preprocess_text(self, text):
#         # Convert text to lowercase
#         text = text.lower()
#         text = ''.join([char for char in text if char.isalnum() or char.isspace()])
#         tokens = word_tokenize(text)
#         stop_words = set(stopwords.words('english'))
#         tokens = [word for word in tokens if word not in stop_words]
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#         return ' '.join(tokens)

#     # def calculate_combined_similarity(self, student_answer, examiner_answer, comprehension, weights):
#     #     # Check if any of the input text strings are empty
#     #     if not student_answer or not examiner_answer or not comprehension:
#     #         return 0.0  # Return zero similarity if any input text string is empty
        
#     #     # Preprocess the text
#     #     preprocessed_student_answer = self.preprocess_text(student_answer)
#     #     preprocessed_examiner_answer = self.preprocess_text(examiner_answer)
#     #     preprocessed_comprehension = self.preprocess_text(comprehension)
        
#     #     # Calculate similarity between student answer and examiner answer
#     #     similarity_examiner = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_examiner_answer))
        
#     #     # Calculate similarity between student answer and comprehension
#     #     similarity_comprehension = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_comprehension))

        
#     #     # Combine similarity scores using weights
#     #     combined_similarity = (weights['examiner'] * similarity_examiner) + (weights['comprehension'] * similarity_comprehension)

#     #     print(combined_similarity)
        
#     #     return combined_similarity


#     def calculate_combined_similarity(self, student_answer, examiner_answer, comprehension, weights=None):
#         # Check if any of the input text strings are empty
#         if not student_answer or not examiner_answer or not comprehension:
#             return 0.0  # Return zero similarity if any input text string is empty
        
#         # Preprocess the text
#         preprocessed_student_answer = self.preprocess_text(student_answer)
#         preprocessed_examiner_answer = self.preprocess_text(examiner_answer)
#         preprocessed_comprehension = self.preprocess_text(comprehension)
        
#         # Calculate similarity between student answer and examiner answer
#         similarity_examiner = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_examiner_answer))
        
#         # Calculate similarity between student answer and comprehension
#         similarity_comprehension = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_comprehension))
        
#         # If no weights provided, determine them dynamically
#         if weights is None:
#             base_examiner = 0.2
#             base_comprehension = 0.8
#             diff = similarity_examiner - similarity_comprehension
            
#             if diff > 0:  # Examiner similarity is higher
#                 shift = min(diff * 0.5, 0.5)  # limit shift
#                 weights = {
#                     'examiner': min(base_examiner + shift, 0.7),
#                     'comprehension': max(base_comprehension - shift, 0.3)
#                 }
#             else:  # Comprehension similarity is higher
#                 shift = min(abs(diff) * 0.5, 0.5)
#                 weights = {
#                     'examiner': max(base_examiner - shift, 0.3),
#                     'comprehension': min(base_comprehension + shift, 0.7)
#                 }

#         # Combine similarity scores using weights
#         combined_similarity = (weights['examiner'] * similarity_examiner) + \
#                             (weights['comprehension'] * similarity_comprehension)

#         print(f"Examiner sim: {similarity_examiner:.3f}, Comprehension sim: {similarity_comprehension:.3f}, "
#             f"Weights: {weights}, Combined: {combined_similarity:.3f}")
        
#         return combined_similarity




#     def predict(self, question_id, comprehension, question, examiner_answer, student_answer, question_score, suppress_warning=True):
#         # Specify weights for examiner answer and comprehension
#         # weights = {'examiner': 0.1, 'comprehension': 0.9}
#         weights = {'examiner': 0.05, 'comprehension': 0.95}
        
#         # Calculate semantic similarity
#         semantic_similarity = self.calculate_combined_similarity(student_answer, examiner_answer, comprehension, weights)
        
#         # Assuming you have loaded your model and preprocessed the text
        
#         # Make prediction using your machine learning model
#         # Concatenate the semantic similarity and question score as features
#         features = np.array([[semantic_similarity, question_score]])  # Reshape the input to a 2D array
#         predicted_student_score = self.model.predict(features)
        
#         # Ensure the predicted score does not exceed the question score
#         predicted_student_score = min(predicted_student_score, question_score)
        
#         return predicted_student_score
















import joblib
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sentence_transformers import SentenceTransformer, util

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

    def refine_with_llm(self, student_answer, examiner_answer, comprehension, raw_score, max_score):
        """Optional: LLM reasoning refinement"""
        prompt = f"""
        You are grading a student answer.
        Comprehension: {comprehension}
        Examiner Answer: {examiner_answer}
        Student Answer: {student_answer}
        Raw Score (from embeddings + keyword overlap): {raw_score:.2f} / {max_score}

        Task: Adjust the score if necessary, considering partial credit.
        Return ONLY a number between 0 and {max_score}.
        """
        response = llm(prompt, max_new_tokens=50, do_sample=False)
        try:
            refined_score = float([t for t in response[0]['generated_text'].split() if t.replace('.', '', 1).isdigit()][0])
            return min(max(refined_score, 0), max_score)  # clamp
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


    # def predict(self, question_id, comprehension, question, examiner_answer, student_answer, question_score):
    #     """Hybrid scoring system"""
    #     # Step 1: Keyword overlap
    #     overlap_count, precision, recall, f1 = self.keyword_overlap(student_answer, examiner_answer)

    #     # Step 2: Semantic similarity
    #     weights = {'examiner': 0.3, 'comprehension': 0.7}  # Adjust per question type if needed
    #     semantic_similarity = self.calculate_combined_similarity(student_answer, examiner_answer, comprehension, weights)

    #     # Step 3: Build feature vector for model
    #     features = np.array([[semantic_similarity, f1, question_score]])
    #     predicted_score = self.model.predict(features)

    #     # Step 4: Rule-based override
    #     if overlap_count >= question_score:
    #         return question_score  # Full marks if student covered all required items

    #     # Step 5: Clip score
    #     # predicted_score = float(min(predicted_score, question_score))
    #     # return predicted_score

    #     predicted_score = float(min(predicted_score, question_score))

    #     # Optional LLM refinement
    #     predicted_score = self.refine_with_llm(student_answer, examiner_answer, comprehension, predicted_score, question_score)

    #     return predicted_score


    def predict(self, question_id, comprehension, question, examiner_answer, student_answer, question_score):
        """Hybrid scoring system with 2-feature input"""
        # Step 1: Keyword overlap
        overlap_count, precision, recall, f1 = self.keyword_overlap(student_answer, examiner_answer)

        # Step 2: Semantic similarity (examiner + comprehension context)
        weights = {'examiner': 0.3, 'comprehension': 0.7}
        semantic_similarity = self.calculate_combined_similarity(
            student_answer, examiner_answer, comprehension, weights
        )

        # Step 3: Build feature vector (ONLY 2 features)
        features = np.array([[semantic_similarity, f1]])

        # Step 4: Model prediction
        predicted_score = self.model.predict(features)[0]

        # Step 5: Rule-based override
        if overlap_count >= question_score:
            return question_score  # Full marks if student covered all required items

        # Step 6: Clip score
        predicted_score = float(min(predicted_score, question_score))

        # Step 7: Optional LLM refinement
        predicted_score = self.refine_with_llm(
            student_answer, examiner_answer, comprehension,
            predicted_score, question_score
        )

        return predicted_score

