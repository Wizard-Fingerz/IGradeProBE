import joblib

class QuestionTypePredictor:
    def __init__(self, model_path='./new_dump/best_model.pkl',
                 vectorizer_path='./new_dump/tfidf_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Question Type model loaded from {model_path}")
        print(f"✅ Vectorizer loaded from {vectorizer_path}")

    def predict(self, question_text: str) -> str:
        """Return the predicted question type as a string"""
        X = self.vectorizer.transform([question_text])
        pred = self.model.predict(X)[0]
        return pred

    def predict_proba(self, question_text: str):
        """Return probability distribution over all categories"""
        X = self.vectorizer.transform([question_text])
        probs = self.model.predict_proba(X)[0]
        return dict(zip(self.model.classes_, probs))
