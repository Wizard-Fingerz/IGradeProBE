import joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings

warnings.simplefilter("ignore")



import joblib
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's medium-sized English language model
nlp = spacy.load("en_core_web_md")

class PredictionService:
    def __init__(self):
        self.model = self.load_model()

    
    def load_model(self):
        model_path = './new_dump/dt_model_new.joblib'

        try:
            model = joblib.load(model_path)
            print('I was succesful')
            return model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None


    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def calculate_combined_similarity(self, student_answer, examiner_answer, comprehension, weights):
        # Check if any of the input text strings are empty
        if not student_answer or not examiner_answer or not comprehension:
            return 0.0  # Return zero similarity if any input text string is empty
        
        # Preprocess the text
        preprocessed_student_answer = self.preprocess_text(student_answer)
        preprocessed_examiner_answer = self.preprocess_text(examiner_answer)
        preprocessed_comprehension = self.preprocess_text(comprehension)
        
        # Calculate similarity between student answer and examiner answer
        similarity_examiner = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_examiner_answer))
        
        # Calculate similarity between student answer and comprehension
        similarity_comprehension = nlp(preprocessed_student_answer).similarity(nlp(preprocessed_comprehension))

        
        # Combine similarity scores using weights
        combined_similarity = (weights['examiner'] * similarity_examiner) + (weights['comprehension'] * similarity_comprehension)

        print(combined_similarity)
        
        return combined_similarity

    def predict(self, question_id, comprehension, question, examiner_answer, student_answer, question_score, suppress_warning=True):
        # Specify weights for examiner answer and comprehension
        # weights = {'examiner': 0.1, 'comprehension': 0.9}
        weights = {'examiner': 0.05, 'comprehension': 0.95}
        
        # Calculate semantic similarity
        semantic_similarity = self.calculate_combined_similarity(student_answer, examiner_answer, comprehension, weights)
        
        # Assuming you have loaded your model and preprocessed the text
        
        # Make prediction using your machine learning model
        # Concatenate the semantic similarity and question score as features
        features = np.array([[semantic_similarity, question_score]])  # Reshape the input to a 2D array
        predicted_student_score = self.model.predict(features)
        
        # Ensure the predicted score does not exceed the question score
        predicted_student_score = min(predicted_student_score, question_score)
        
        return predicted_student_score
