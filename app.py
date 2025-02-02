import re
import logging
import numpy as np
from urllib.parse import urlparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of phishing-related keywords
phishing_keywords = ["login", "email", "verify", "account", "update"]

class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features based on the presence of specific keywords in a URL.
    """
    def __init__(self, keywords):
        self.keywords = keywords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        keyword_features = []
        for url in X:
            features = [int(keyword in url) for keyword in self.keywords]
            keyword_features.append(features)
        return np.array(keyword_features)

class AdditionalURLFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts additional features such as domain length, path length,
    presence of an IP address, special character count, and entropy.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for url in X:
            if not url:
                features.append([0, 0, 0, 0, 0])  # Default values for empty URLs
                continue
            try:
                parsed_url = urlparse(url)
                domain_length = len(parsed_url.netloc)
                path_length = len(parsed_url.path)
                has_ip = int(bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', parsed_url.netloc)))
                special_chars_count = sum(1 for char in url if char in "-_@#")
                entropy = -sum(url.count(c) / len(url) * np.log2(url.count(c) / len(url)) for c in set(url))
                features.append([domain_length, path_length, has_ip, special_chars_count, entropy])
            except Exception as e:
                logging.warning("Error processing URL %s: %s", url, e)
                features.append([0, 0, 0, 0, 0])
        return np.array(features)

# Function to check if a given URL is valid
def is_valid_url(url):
    """Checks if the URL follows a standard format."""
    url_pattern = re.compile(r'^(https?:\/\/)?([a-zA-Z0-9-_]+\.)+[a-zA-Z]{2,6}\/?')
    return bool(url_pattern.match(url))

# Load or train a simple classifier to classify URLs as phishing or legitimate
def load_or_train_model():
    model_path = 'url_classifier.pkl'
    try:
        model = joblib.load(model_path)
        logging.info("Loaded pre-trained model.")
    except FileNotFoundError:
        logging.info("Training new model.")
        # Dummy dataset with sample features and labels
        X_train = np.random.rand(100, 5)  # Replace with real extracted features
        y_train = np.random.choice([0, 1], size=100)  # 0: Legitimate, 1: Phishing
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        logging.info("Model trained and saved.")
    return model

model = load_or_train_model()

def classify_url(url):
    """Classifies a given URL as phishing or legitimate."""
    if not is_valid_url(url):
        return "Invalid URL format"
    
    keyword_features = KeywordFeatureExtractor(phishing_keywords).transform([url])
    additional_features = AdditionalURLFeatures().transform([url])
    features = np.hstack((keyword_features, additional_features))
    prediction = model.predict(features)
    return "Phishing URL" if prediction[0] == 1 else "Legitimate URL"
