
from model import PhishingModel
import numpy as np

def verify():
    print("Initializing model...")
    m = PhishingModel()
    
    test_cases = [
        # Email Text Cases
        ("URGENT: Verify your bank account immediately to avoid suspension http://bit.ly/fake", "Phishing"),
        ("Dear user, you have won a lottery. Click here to claim your prize.", "Phishing"),
        ("Hey, are we still meeting for lunch tomorrow?", "Safe"),
        ("Please find the attached invoice for your reference.", "Safe"),
        # URL Cases
        ("http://paypal-secure-login.com", "Phishing"),
        ("https://google.com", "Safe"),
        ("http://192.168.1.1/login.php", "Phishing"), # IP address often suspicious in some contexts, but let's see model behavior
        ("https://www.youtube.com", "Safe"),
        ("http://verify-account-security-update.net", "Phishing")
    ]
    
    print("\n--- Running Verification ---")
    for text, expected in test_cases:
        is_phishing, confidence = m.predict(text)
        result_str = "Phishing" if is_phishing else "Safe"
        status = "PASS" if result_str == expected else "FAIL"
        
        # Convert numpy types to native for cleaner printing if needed
        if isinstance(confidence, (np.floating, float)):
             conf_str = f"{confidence:.2f}"
        else:
             conf_str = str(confidence)

        print(f"[{status}] Text: {text[:40]}... | Prediction: {result_str} ({conf_str}) | Expected: {expected}")

if __name__ == "__main__":
    verify()
