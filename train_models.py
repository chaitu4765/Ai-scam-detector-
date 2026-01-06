import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

def extract_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_at'] = url.count('@')
    features['num_params'] = url.count('?')
    features['num_segments'] = url.count('/')
    features['is_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', url) else 0
    return features

def train_url_model():
    print("Training URL Model (using subset for speed)...")
    df = pd.read_csv('d:/PHISHING SITE/archive/phishing_site_urls.csv')
    df = df.sample(50000, random_state=42) # Subset for performance
    
    # Feature extraction for URLs
    url_features = df['URL'].apply(extract_url_features)
    feature_df = pd.DataFrame(url_features.tolist())
    
    # Text-based features for URLs (TF-IDF on characters/n-grams)
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
    X_tfidf = tfidf.fit_transform(df['URL'])
    
    # Combine features using scipy.sparse.hstack
    from scipy.sparse import hstack
    X = hstack([X_tfidf, feature_df.values])
    y = df['Label'].map({'good': 0, 'bad': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"URL Model Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/url_model.pkl')
    joblib.dump(tfidf, 'models/url_tfidf.pkl')
    print("URL models saved.")

def train_email_model():
    print("Training Email Model (aggressive fix for friendly emails)...")
    df = pd.read_csv('d:/PHISHING SITE/phishing  dataset email/phishing_email.csv')
    df = df.dropna(subset=['text_combined', 'label'])
    
    # 1. Expanded "friendly" messages to reduce false positives
    friendly_messages = [
        "Hey, how are you?", "See you tomorrow for lunch.", "The meeting is at 2 PM.",
        "Thanks for the help!", "Can you send me that file?", "Let's catch up soon.",
        "Happy birthday!", "Are we still meeting today?", "I'll be there in 5 minutes.",
        "Good morning, hope you have a great day!", "Thank you for the update.",
        "I'm running late, starting in 10.", "hey there! how are you", "just checking in",
        "Hi, let me know when you're free to chat.", "Dinner at 7?", "Got it, thanks!",
        "Can we reschedule?", "Nice to meet you!", "Looking forward to it.",
        "What's up?", "How's your day going?", "Talk to you later.",
        "Yes, that sounds good.", "No problem at all.", "See ya!",
        "Any updates on the project?", "Could you please take a look?",
        "Have a great weekend!", "Welcome back!", "Hope you slept well."
    ]
    
    # 2. Massive Oversampling (500x) to ensure these patterns dominate legitimate classification
    oversampled_friendly = friendly_messages * 500
    friendly_df = pd.DataFrame({'text_combined': oversampled_friendly, 'label': [0] * len(oversampled_friendly)})
    
    # Combine with main dataset
    df = pd.concat([df, friendly_df], ignore_index=True)
    
    # Increase sample size to keep enough real data while dominated by synthetic legit for phrases
    df = df.sample(min(60000, len(df)), random_state=42)
    
    # 3. Improved TF-IDF: ngram_range (1, 3) captures full phrases, min_df wipes noise
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, min_df=2)
    X = tfidf.fit_transform(df['text_combined'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Tuned Model: Lower C (stronger regularization) to prevent overfitting to specific noisy words
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Email Model Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/email_model.pkl')
    joblib.dump(tfidf, 'models/email_tfidf.pkl')
    print("Email models saved.")

if __name__ == "__main__":
    train_url_model()
    train_email_model()
