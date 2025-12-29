import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import re

def tokenize_url(url):
    """Custom tokenizer to split URLs into meaningful tokens."""
    # Split by slash, dot, hyphen, underscore, etc.
    tokens = re.split(r'[.,/\-_%?&=]', url)
    # Remove empty strings
    return [t for t in tokens if t]

def train_url_model():
    try:
        print("Loading URL dataset...")
        dataset_path = r"../archive/phishing_site_urls.csv"
        
        if not os.path.exists(dataset_path):
             dataset_path = r"archive/phishing_site_urls.csv"
             
        if not os.path.exists(dataset_path):
             dataset_path = r"c:\Users\LAVANYA\Downloads\new project\archive\phishing_site_urls.csv"
             
        print(f"Reading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Preprocessing
        print("Preprocessing...")
        # Map labels: 'bad' -> 1 (Phishing), 'good' -> 0 (Safe)
        # Assuming values are 'bad' and 'good' based on inspection
        # Check unique values first if unsure, but standard for this dataset
        df['label_num'] = df['Label'].map({'bad': 1, 'good': 0})
        
        # Drop any failed mappings (though unlikely if clean)
        df = df.dropna(subset=['label_num'])
        
        X = df['URL']
        y = df['label_num']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training URL model (Logistic Regression)...")
        # Logistic Regression is usually faster and very effective for high-dim sparse features like URL tokens
        model = make_pipeline(
            TfidfVectorizer(tokenizer=tokenize_url, token_pattern=None, max_features=50000), 
            LogisticRegression(max_iter=1000)
        )
        model.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        output_path = 'url_model.pkl'
        print(f"Saving model to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
            
        print("URL Training complete!")

    except Exception as e:
        print(f"Error during URL training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_url_model()
