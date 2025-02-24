import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

feedback_file = "Customer Feedback Analysis.csv"
segmentation_file = "Customer Segmentation for Personalized Marketing.csv"

df_feedback = pd.read_csv(feedback_file).dropna(subset=["Reviews"]) 
df_segmentation = pd.read_csv(segmentation_file, encoding="latin1")


# Preprocess text function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        return ' '.join(text.split())
    return ""

df_feedback["Processed_Reviews"] = df_feedback["Reviews"].apply(clean_text)

# Categorization function using VADER
def classify_text(text):
    sentiment = sia.polarity_scores(text)

    if sentiment["compound"] <= -0.3:  # Negative sentiment → Complaint
        return "Complaint"
    elif -0.3 < sentiment["compound"] < 0.3:  # Neutral sentiment → Suggestion
        return "Suggestion"
    else:  # Positive sentiment → Praise
        return "Praise"

df_feedback["Category"] = df_feedback["Processed_Reviews"].apply(classify_text)
df_filtered = df_feedback[df_feedback["Category"] != "Neutral"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_filtered["Processed_Reviews"], df_filtered["Category"], test_size=0.2, random_state=42)

# Train model
model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, "feedback_classifier.pkl")