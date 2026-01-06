# Phishing & Scam Detection AI

A web-based application that uses machine learning to detect phishing URLs and scam emails.

## Features
- **URL Detection**: Analyzes URLs to determine if they are legitimate or phishing.
- **Email/Text Detection**: Scans message content for common scam patterns.
- **Real-time Prediction**: Fast, local API for immediate results.
- **Modern UI**: Clean, responsive interface with interactive elements.

## Project Structure
```text
PHISHING SITE/
├── app/
│   ├── backend/        # Flask API
│   └── frontend/       # HTML/CSS/JS interface
├── models/             # Trained ML models and vectorizers
├── training/           # Scripts for model training (train_models.py)
└── datasets/           # Raw data used for training
```

## Local Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone or download the repository.
2. Install dependencies:
   ```bash
   pip install flask flask-cors joblib pandas numpy scipy scikit-learn
   ```

### Running the Application
1. **Start the Backend**:
   ```bash
   cd "d:/PHISHING SITE"
   python app/backend/app.py
   ```
   The backend will start on `http://127.0.0.1:5000`.

2. **Launch the Frontend**:
   Simply open `app/frontend/index.html` in any modern web browser.

## Usage
- **To scan a URL**: Enter the suspicious URL in the search bar and click "Detect Scam".
- **To scan an Email**: Paste the email content into the message analyzer tool.

## Technical Details
- **Backend**: Flask with Scikit-learn models.
- **ML Models**: Bag-of-words (TF-IDF) combined with custom URL features.
- **Frontend**: Vanilla JavaScript and CSS for high performance.
