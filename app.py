import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st
import random

def train_model():
    true_data = pd.read_csv("true.csv")
    true_data['label'] = 0

    fake_data = pd.read_csv("fake.csv")
    fake_data['label'] = 1

    data = pd.concat([true_data, fake_data], ignore_index=True)
    data = data[['text', 'label']].dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(120deg, #1f1c2c, #928DAB);
            color: #f4f4f4;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            color: #FFD700;
            text-align: center;
        }
        .subtitle {
            font-size: 1.2rem;
            font-weight: 400;
            text-align: center;
            color: #ffffffcc;
            margin-bottom: 2rem;
        }
        div.stButton > button:first-child {
            background-color: #00C9A7;
            color: black;
            font-weight: 700;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1.1rem;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #00B189;
            color: white;
        }
        textarea {
            border-radius: 10px !important;
            font-size: 1rem;
            padding: 15px !important;
            background-color: #2e2d3b;
            color: #f1f1f1 !important;
        }
        .css-1d391kg {
            background-color: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
        }
        footer {
            font-size: 0.8rem;
            color: #ccc;
            text-align: center;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Paste a news article below to check if it's Real or Fake.</p>", unsafe_allow_html=True)

    # Session state to hold the input
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""

    try:
        true_data = pd.read_csv("true.csv")
        fake_data = pd.read_csv("fake.csv")
        total = len(true_data) + len(fake_data)

        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Real News Samples", len(true_data))
        col2.metric("🔴 Fake News Samples", len(fake_data))
        col3.metric("📄 Total Records", total)

        if st.button("🎲 Try Sample News"):
            all_data = pd.concat([true_data, fake_data], ignore_index=True)
            sample = all_data.sample(n=1).iloc[0]
            st.session_state.news_input = sample['text']
            st.info("📝 Sample News Loaded!")

    except:
        st.warning("Dataset preview not available (CSV files missing or unreadable).")

    input_text = st.text_area("Enter News Article", value=st.session_state.news_input, height=250, placeholder="Paste the news text here...")

    if st.button("Predict"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some news text before clicking Predict.")
        else:
            st.session_state.news_input = input_text
            model, tfidf = load_model()
            vector = tfidf.transform([input_text])
            prediction = model.predict(vector)[0]
            if prediction == 0:
                st.success("🟢 Prediction: This news is **Real**.")
            else:
                st.error("🔴 Prediction: This news is **Fake**.")

    st.sidebar.header("📌 About")
    st.sidebar.info(
        """
        This app uses a Logistic Regression model trained on a dataset of real and fake news.

        Enter any news article in the text box to see whether it is classified as Real or Fake.

        Built with Python, Scikit-learn & Streamlit.
        """
    )

    st.markdown("<footer>Made with ❤️ by Alekhya | Logistic Regression + Streamlit</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Uncomment this once to train your model locally, then comment again
    # train_model()
    main()
