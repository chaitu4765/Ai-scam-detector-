import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train():
    try:
        # 1. Load Data
        print("Loading data...")
        dataset_path = r"../phishing  dataset email/phishing_email.csv"
        
        if not os.path.exists(dataset_path):
             # Fallback for different CWD
             dataset_path = r"phishing  dataset email/phishing_email.csv"
             
        if not os.path.exists(dataset_path):
             # Try absolute path based on known user structure if relative fails
             dataset_path = r"c:\Users\LAVANYA\Downloads\new project\phishing  dataset email\phishing_email.csv"

        print(f"Reading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # 2. Preprocessing
        print("Preprocessing...")
        # Inspect columns - we expect 'text_combined' and 'label'
        if 'text_combined' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Dataset columns mismatch. Found: {df.columns}")

        # Drop NaNs
        df = df.dropna(subset=['text_combined', 'label'])
        
        X = df['text_combined']
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train Model
        print("Training model (this might take a moment)...")
        # TF-IDF + Naive Bayes is a strong baseline for text classification
        model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=50000), MultinomialNB())
        model.fit(X_train, y_train)

        # 4. Evaluate
        print("Evaluating...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # 5. Save Model
        output_path = 'phishing_model.pkl'
        print(f"Saving model to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print("Training complete!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
