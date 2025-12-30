import pickle
import os
import re

# Need to define tokenizer for pickle loading if it's not in the main scope, 
# but usually it's safer to have it here or import it.
def tokenize_url(url):
    tokens = re.split(r'[.,/\-_%?&=]', url)
    return [t for t in tokens if t]

class PhishingModel:
    def __init__(self, email_model_path="phishing_model.pkl", url_model_path="url_model.pkl"):
        self.email_model = None
        self.url_model = None
        self.errors = []
        
        # Get the absolute directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        print(f"Model loader initialized. Current dir: {current_dir}, Root dir: {root_dir}")
        
        def find_model_path(filename):
            paths_to_check = [
                os.path.join(current_dir, filename),
                os.path.join(root_dir, filename),
                os.path.join(os.getcwd(), filename),
                os.path.join(os.getcwd(), 'api', filename)
            ]
            for path in paths_to_check:
                if os.path.exists(path):
                    return path
            return None

        # Load Email Model
        try:
            full_email_path = find_model_path(email_model_path)
            if full_email_path:
                print(f"Loading email model from: {full_email_path}")
                with open(full_email_path, 'rb') as f:
                    self.email_model = pickle.load(f)
                print("Email phishing model loaded successfully.")
            else:
                err = f"Email model {email_model_path} not found."
                print(err)
                self.errors.append(err)
        except Exception as e:
             err = f"Email model load error: {e}"
             print(err)
             self.errors.append(err)

        # Load URL Model
        try:
            full_url_path = find_model_path(url_model_path)
            if full_url_path:
                print(f"Loading URL model from: {full_url_path}")
                with open(full_url_path, 'rb') as f:
                    self.url_model = pickle.load(f)
                print("URL phishing model loaded successfully.")
            else:
                err = f"URL model {url_model_path} not found."
                print(err)
                self.errors.append(err)
        except Exception as e:
             err = f"URL model load error: {e}"
             print(err)
             self.errors.append(err)

    def predict_url(self, url):
        if self.url_model:
            try:
                # 1 = Phishing (Bad), 0 = Safe (Good)
                prediction = self.url_model.predict([url])[0]
                proba = self.url_model.predict_proba([url])[0]
                # If prediction is 1 (Phishing), confidence is proba[1]
                # If prediction is 0 (Safe), confidence is proba[0]
                confidence = proba[prediction]
                return prediction == 1, confidence
            except Exception as e:
                print(f"URL prediction error: {e}")
                pass
        return False, 0.0

    def predict(self, text):
        # 1. Check if text is just a URL
        # Simple regex to check if it looks like a URL and nothing else significant
        url_pattern = r'^(http|https)://[^\s]+$'
        if re.match(url_pattern, text.strip()) or (text.strip().startswith("www.") and len(text.split()) == 1):
            print("Input detected as URL.")
            return self.predict_url(text.strip())

        # 2. Otherwise use Email Text model
        if self.email_model:
            try:
                prediction = self.email_model.predict([text])[0]
                if hasattr(self.email_model, "predict_proba"):
                     proba = self.email_model.predict_proba([text])[0]
                     # Return confidence in the predicted class
                     confidence = proba[prediction] 
                     return prediction == 1, confidence
                return prediction == 1, 1.0
            except Exception as e:
                print(f"Prediction error: {e}")
                pass
        
        # Mock Fallbacks...
        return False, 0.0

model = PhishingModel()
