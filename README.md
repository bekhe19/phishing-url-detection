URL Phishing Detection
This project is a machine-learning-based URL classifier that detects phishing websites. It extracts key features from a given URL, including keyword presence, domain structure, and entropy, to classify it as either phishing or legitimate. The model is trained using a RandomForestClassifier and can be updated with new data.

Features:
Extracts keyword-based and structural features from URLs
Uses a pre-trained or dynamically trained RandomForest model
Supports new URL classification using extracted features
Usage:
Simply input a URL, and the model will predict whether it's phishing or legitimate.
