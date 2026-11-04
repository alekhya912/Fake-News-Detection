# Fake News Detection App
A Machine Learning-based web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Logistic Regression.

Project Overview
The Fake News Detection App uses a machine learning model trained on news datasets to identify potentially fake news. It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for classification.
The application provides a simple and interactive interface built with Streamlit, allowing users to enter news content and receive real-time predictions.

Features
- Classifies news as Real or Fake.
- Uses TF-IDF for text feature extraction.
- Logistic Regression for machine learning classification.
- Interactive web interface using Streamlit.
- Real-time predictions based on user-inputted news content.

Technologies Used
- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Algorithm:** Logistic Regression
- **NLP Technique:** TF-IDF Vectorization
- **Web Framework:** Streamlit
- **Dataset:** Kaggle

## 📂 Project Structure
```
Fake-News-Detection/
│
├── app.py
├── true.csv
├── fake.csv
├── requirements.txt
└── README.md
```
How It Works
1. The user enters a news article or headline into the application.
2. The input text is processed using TF-IDF vectorization.
3. The Logistic Regression model analyzes the extracted text features.
4. The application displays the prediction as **Real News** or **Fake News**.

Dataset
The project uses a news dataset obtained from Kaggle, containing real and fake news articles for training and evaluation.

Future Enhancements
- Improve model performance using additional machine learning algorithms.
- Add model evaluation metrics such as accuracy, precision, recall and F1-score.
- Enhance the user interface.
- Deploy the application online for public access.

