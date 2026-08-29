from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



# NLTK Resources
# =========================

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")


# Text Preprocessing
# =========================

ps = PorterStemmer()


def transform_text(text):

    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# FASTAPI APP
# =========================

app = FastAPI(
    title="Spam SMS Classifier",
    description="Machine Learning based Spam SMS Detection API",
    version="1.0.0"
)


# LOAD MODEL AND VECTORIZER
# =========================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    return html_content



# PREDICTION API
# =========================

@app.post("/predict")
def predict(message: str = Form(...)):

    # 1. Preprocess SMS
    transformed_text = transform_text(message)

    # 2. Convert text into TF-IDF
    vector_input = tfidf.transform([transformed_text])

    # 3. Prediction
    prediction = model.predict(vector_input)[0]

    # 4. Probability
    probabilities = model.predict_proba(vector_input)[0]

    # 5. Get Class Labels
    classes = model.classes_

    spam_probability = (
        probabilities[list(classes).index(1)] * 100
    )

    not_spam_probability = (
        probabilities[list(classes).index(0)] * 100
    )

    # 6. Final Result
    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    # 7. Return Response
    return {
        "message": message,
        "prediction": result,
        "spam_probability": round(spam_probability, 2),
        "not_spam_probability": round(not_spam_probability, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
