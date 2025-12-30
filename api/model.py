import pickle
import os
import re
import math


def tokenize_url(url):
    tokens = re.split(r'[.,/\-_%?&=]', url)
    return [t for t in tokens if t]


def boost_confidence(raw_confidence, is_phishing):
    """
    Boosts confidence in a controlled & explainable way.
    Clamps between 0.55 and 0.99.
    """
    # 1. Confidence floor
    MIN_CONFIDENCE = 0.65 if is_phishing else 0.55

    # 2. Sigmoid stretching (center at 0.5, stretch with multiplier)
    stretched = 1 / (1 + math.exp(-8 * (raw_confidence - 0.5)))

    # 3. Weighted blend
    boosted = (0.7 * stretched) + (0.3 * raw_confidence)

    # 4. Apply floor and strict clamp per requirements
    boosted = max(boosted, MIN_CONFIDENCE)
    
    # 5. Final clamp strictly between 0.55 and 0.99
    final_conf = max(0.55, min(boosted, 0.99))
    
    return round(final_conf, 4)


class PhishingModel:
    def __init__(self, email_model_path="phishing_model.pkl", url_model_path="url_model.pkl"):
        self.email_model = None
        self.url_model = None
        self.errors = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)

        def find_model_path(filename):
            for path in [
                os.path.join(current_dir, filename),
                os.path.join(root_dir, filename),
                os.path.join(os.getcwd(), filename),
                os.path.join(os.getcwd(), 'api', filename)
            ]:
                if os.path.exists(path):
                    return path
            return None

        try:
            path = find_model_path(email_model_path)
            if path:
                with open(path, 'rb') as f:
                    self.email_model = pickle.load(f)
        except Exception as e:
            self.errors.append(str(e))

        try:
            path = find_model_path(url_model_path)
            if path:
                with open(path, 'rb') as f:
                    self.url_model = pickle.load(f)
        except Exception as e:
            self.errors.append(str(e))

    def predict_url(self, url):
        if not self.url_model:
            return False, 0.0

        try:
            prediction = self.url_model.predict([url])[0]
            proba = self.url_model.predict_proba([url])[0]

            raw_confidence = proba[prediction]
            is_phishing = prediction == 1

            confidence = boost_confidence(raw_confidence, is_phishing)

            return is_phishing, confidence

        except Exception as e:
            print(f"URL prediction error: {e}")
            return False, 0.0

    def predict(self, text):
        url_pattern = r'^(http|https)://[^\s]+$'

        if re.match(url_pattern, text.strip()) or (
            text.strip().startswith("www.") and len(text.split()) == 1
        ):
            return self.predict_url(text.strip())

        if not self.email_model:
            return False, 0.0

        try:
            prediction = self.email_model.predict([text])[0]

            if hasattr(self.email_model, "predict_proba"):
                proba = self.email_model.predict_proba([text])[0]
                raw_confidence = proba[prediction]
            else:
                raw_confidence = 0.75  # fallback assumption

            is_phishing = prediction == 1
            confidence = boost_confidence(raw_confidence, is_phishing)

            return is_phishing, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0


model = PhishingModel()
