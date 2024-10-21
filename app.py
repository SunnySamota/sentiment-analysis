import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords
import nltk
nltk.download('stopwords')

# Load the saved model and vectorizer
model = pickle.load(open('twitter_sentiment_analysis.sav', 'rb'))

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function for text preprocessing (stemming)
def preprocess_text(content):
    # Clean text using regex
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    # Remove stopwords and apply stemming
    stemmed_content = [stemmer.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load the TfidfVectorizer (if it's saved separately)
vectorizer = TfidfVectorizer()
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure this file exists

# Streamlit app UI
st.title("Twitter Sentiment Analysis App")
st.write("Enter a tweet or sentence to analyze its sentiment.")

# Text input from the user
user_input = st.text_area("Your Text", "")

if st.button("Analyze"):
    if user_input:
        # Preprocess the user input
        preprocessed_text = preprocess_text(user_input)

        # Vectorize the input using the loaded vectorizer
        input_vector = vectorizer.transform([preprocessed_text])

        # Predict the sentiment using the loaded model
        prediction = model.predict(input_vector)[0]

        # Display the result
        if prediction == 1:
            st.success("The sentiment is **Positive** 😊")
        else:
            st.error("The sentiment is **Negative** 😢")
    else:
        st.warning("Please enter some text to analyze.")



  