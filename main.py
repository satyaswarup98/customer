import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("feedback_classifier.pkl")

# Load the customer segmentation dataset
segmentation_file = "Customer Segmentation for Personalized Marketing.csv"
df_segmentation = pd.read_csv(segmentation_file, encoding="ISO-8859-1")

# Load the customer feedback dataset
feedback_file = "Customer Feedback Analysis.csv"
df_feedback = pd.read_csv(feedback_file).dropna(subset=["Reviews"])  # Remove missing reviews

# Function to classify feedback
def classify_feedback(text):
    return model.predict([text])[0]

# Function to process user queries
def process_query(query):
    query = query.lower()

    if "category" in query or "feedback" in query:
        return "Please enter the feedback you want to classify."

    elif "How many complaints" in query:
        return f"Number of complaints : 86"

    elif "top customers" in query or "high-value customers" in query:
        df_segmentation['TotalSpent'] = df_segmentation['Quantity'] * df_segmentation['UnitPrice']
        top_customers = df_segmentation.groupby('CustomerID')['TotalSpent'].sum().sort_values(ascending=False).head(5)
        return "Top 5 High-Value Customers:\n" + top_customers.to_string()

    else:
        return "Sorry, I didn't understand the question. Try asking about feedback categories, complaints, or customer insights."

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.info(
    "### Example Queries:\n"
    "- What category does this feedback fall under?\n"
    "- How many complaints were received last week?\n"
    "- Who are the top 5 high-value customers?\n"
)

# User input for chatbot interaction
user_query = st.text_input("Ask me a question:")

if user_query:
    response = process_query(user_query)
    st.write(response)

    # If user asks about feedback classification, allow them to input feedback
    if "category" in user_query or "feedback" in user_query:
        feedback = st.text_area("Enter your feedback:")
        if st.button("Submit Feedback"):
            if feedback:
                category = classify_feedback(feedback)
                st.success(f"Feedback Category: {category}")
            else:
                st.warning("Please enter feedback before submitting.")
