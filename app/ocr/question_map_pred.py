import joblib
import numpy as np

class SVMQuestionMappingPredictor:
    """
    Predicts whether a given sentence is a 'question' or an 'answer'
    using a pre-trained SVM classification model and vectorizer.
    Compatible with label encoders for robust output.
    """
    def __init__(
        self,
        model_path='./new_dump/svmModel.joblib',
        vectorizer_path='./new_dump/Xvectorizer.joblib',
        label_encoder_path='./new_dump/Xlabel_encoder.joblib.joblib'
    ):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        try:
            self.label_encoder = joblib.load(label_encoder_path)
            self.has_label_encoder = True
            print(f"✅ SVM label encoder loaded from {label_encoder_path}")
        except Exception:
            self.label_encoder = None
            self.has_label_encoder = False
            print("⚠️ No label encoder found, using model.classes_ as labels.")
        print(f"✅ SVM model loaded from {model_path}")
        print(f"✅ SVM vectorizer loaded from {vectorizer_path}")

    def predict_category(self, sentence: str) -> str:
        X = self.vectorizer.transform([sentence])
        pred = self.model.predict(X)[0]
        if self.has_label_encoder:
            # pred is int, decode to string
            return self.label_encoder.inverse_transform([pred])[0]
        else:
            # pred is label string
            return pred

    def predict_category_proba(self, sentence: str):
        X = self.vectorizer.transform([sentence])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
        elif hasattr(self.model, "_predict_proba_lr"):
            probs = self.model._predict_proba_lr(X)[0]
        else:
            pred = self.model.predict(X)[0]
            classes = self.model.classes_
            one_hot = [1.0 if c == pred else 0.0 for c in classes]
            probs = np.array(one_hot)
        # Get class labels
        if self.has_label_encoder:
            # model.classes_ are ints, decode to strings
            labels = self.label_encoder.inverse_transform(self.model.classes_)
        else:
            labels = self.model.classes_
        return dict(zip(labels, probs))


class XGBQuestionMappingPredictor:
    """
    Predicts whether a given sentence is a 'question' or an 'answer'
    using a pre-trained XGBoost classification model and vectorizer.
    Compatible with label encoders for robust output.
    """
    def __init__(
        self,
        model_path='./new_dump/Xgb_model.joblib',
        vectorizer_path='./new_dump/Xvectorizer.joblib',
        label_encoder_path='./new_dump/Xlabel_encoder.joblib'
    ):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        try:
            self.label_encoder = joblib.load(label_encoder_path)
            self.has_label_encoder = True
            print(f"✅ XGBoost label encoder loaded from {label_encoder_path}")
        except Exception:
            self.label_encoder = None
            self.has_label_encoder = False
            print("⚠️ No label encoder found for XGBoost, using model.classes_ as labels.")
        print(f"✅ XGBoost model loaded from {model_path}")
        print(f"✅ XGBoost vectorizer loaded from {vectorizer_path}")

    def predict_category(self, sentence: str) -> str:
        X = self.vectorizer.transform([sentence])
        pred = self.model.predict(X)[0]
        if self.has_label_encoder:
            # pred is int, decode to string
            return self.label_encoder.inverse_transform([int(pred)])[0]
        else:
            # pred is label string or int
            return str(pred)

    def predict_category_proba(self, sentence: str):
        X = self.vectorizer.transform([sentence])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
        else:
            # fallback: one-hot
            pred = self.model.predict(X)[0]
            if hasattr(self.model, "classes_"):
                classes = self.model.classes_
            else:
                # fallback: use label encoder classes if available
                if self.has_label_encoder:
                    classes = np.arange(len(self.label_encoder.classes_))
                else:
                    raise RuntimeError("Cannot determine class labels for XGBoost model.")
            one_hot = [1.0 if c == pred else 0.0 for c in classes]
            probs = np.array(one_hot)
        # Get class labels
        if self.has_label_encoder:
            if hasattr(self.model, "classes_"):
                labels = self.label_encoder.inverse_transform(self.model.classes_)
            else:
                labels = self.label_encoder.classes_
        elif hasattr(self.model, "classes_"):
            labels = self.model.classes_
        else:
            labels = [str(i) for i in range(len(probs))]
        return dict(zip(labels, probs))


class BERTQuestionMappingPredictor:
    """
    Predicts whether a given sentence is a 'question' or an 'answer'
    using a pre-trained BERT model and tokenizer.
    """
    def __init__(self, model_path='./new_dump/bert_model.joblib',
                 tokenizer_path='./new_dump/tokenizer.joblib',
                 label_encoder_path='./new_dump/bert_label_encoder.joblib'):
        # Load the pre-trained BERT model and tokenizer from disk
        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        try:
            self.label_encoder = joblib.load(label_encoder_path)
            self.has_label_encoder = True
            print(f"✅ BERT label encoder loaded from {label_encoder_path}")
        except Exception:
            self.label_encoder = None
            self.has_label_encoder = False
            print("⚠️ No label encoder found for BERT, using id2label or indices.")
        print(f"✅ BERT model loaded from {model_path}")
        print(f"✅ BERT tokenizer loaded from {tokenizer_path}")

    def predict_category(self, sentence: str) -> str:
        """
        Predict whether the sentence is a 'question' or an 'answer'.
        Returns the predicted label.
        """
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        import torch
        with torch.no_grad():
            if isinstance(self.model, dict):
                raise TypeError("Loaded BERT model is a dict, not a callable model. Please check your model serialization.")
            elif callable(self.model):
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            elif hasattr(self.model, "predict"):
                outputs = self.model.predict(inputs)
                logits = outputs
            else:
                raise TypeError("Loaded BERT model is not callable. Please check your model serialization.")
            predicted_class = logits.argmax(dim=1).item()
            # Try label encoder first, then id2label, then index as string
            if self.has_label_encoder:
                return self.label_encoder.inverse_transform([predicted_class])[0]
            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                return self.model.config.id2label[predicted_class]
            return str(predicted_class)

    def predict_category_proba(self, sentence: str):
        """
        Get the probability distribution over the possible categories
        for a sentence.
        Returns a dict mapping label to probability.
        """
        import torch
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            if isinstance(self.model, dict):
                raise TypeError("Loaded BERT model is a dict, not a callable model. Please check your model serialization.")
            elif callable(self.model):
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            elif hasattr(self.model, "predict_proba"):
                outputs = self.model.predict_proba(inputs)
                logits = outputs
            else:
                raise TypeError("Loaded BERT model is not callable. Please check your model serialization.")
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            # Try label encoder, then id2label, then indices
            if self.has_label_encoder:
                labels = self.label_encoder.inverse_transform(np.arange(len(probs)))
            elif hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                labels = [self.model.config.id2label[i] for i in range(len(probs))]
            else:
                labels = [str(i) for i in range(len(probs))]
            return dict(zip(labels, probs))


if __name__ == "__main__":
    # SVM Example
    print("=== SVM Predictor ===")
    svm_predictor = SVMQuestionMappingPredictor()
    test_sentences = [
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Explain the process of photosynthesis.",
        "Photosynthesis is the process by which green plants convert sunlight into energy."
    ]
    for test_sentence in test_sentences:
        predicted_category = svm_predictor.predict_category(test_sentence)
        print(f"Input: {test_sentence}")
        print(f"Predicted label: {predicted_category}")
        proba = svm_predictor.predict_category_proba(test_sentence)
        print("Category probabilities:")
        for cat, p in proba.items():
            print(f"  {cat}: {p:.4f}")
        print("-" * 40)

    # XGBoost Example
    print("=== XGBoost Predictor ===")
    try:
        xgb_predictor = XGBQuestionMappingPredictor()
        for test_sentence in test_sentences:
            predicted_category = xgb_predictor.predict_category(test_sentence)
            print(f"Input: {test_sentence}")
            print(f"Predicted label: {predicted_category}")
            proba = xgb_predictor.predict_category_proba(test_sentence)
            print("Category probabilities:")
            for cat, p in proba.items():
                print(f"  {cat}: {p:.4f}")
            print("-" * 40)
    except Exception as e:
        print("XGBoost predictor could not be run (missing model/vectorizer/label encoder):", e)

    # BERT Example (requires torch, transformers, and compatible joblib dumps)
    try:
        import torch
        print("=== BERT Predictor ===")
        bert_predictor = BERTQuestionMappingPredictor()
        for test_sentence in test_sentences:
            predicted_category = bert_predictor.predict_category(test_sentence)
            print(f"Input: {test_sentence}")
            print(f"Predicted label: {predicted_category}")
            proba = bert_predictor.predict_category_proba(test_sentence)
            print("Category probabilities:")
            for cat, p in proba.items():
                print(f"  {cat}: {p:.4f}")
            print("-" * 40)
    except Exception as e:
        print("BERT predictor could not be run (missing torch/transformers or model/tokenizer):", e)
