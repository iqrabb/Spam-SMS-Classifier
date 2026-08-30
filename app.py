from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import pickle
import re
from nltk.stem.porter import PorterStemmer


# Text Preprocessing


ps = PorterStemmer()

# Hardcoded English stopwords list (no NLTK download required)
ENGLISH_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","you're","you've",
    "you'll","you'd","your","yours","yourself","yourselves","he","him","his",
    "himself","she","she's","her","hers","herself","it","it's","its","itself",
    "they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","that'll","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a",
    "an","the","and","but","if","or","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very",
    "s","t","can","will","just","don","don't","should","should've","now","d",
    "ll","m","o","re","ve","y","ain","aren","aren't","couldn","couldn't",
    "didn","didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't","haven",
    "haven't","isn","isn't","ma","mightn","mightn't","mustn","mustn't","needn",
    "needn't","shan","shan't","shouldn","shouldn't","wasn","wasn't","weren",
    "weren't","won","won't","wouldn","wouldn't"
}


def transform_text(text):

    text = text.lower()

    # Tokenization (regex based — no NLTK download needed)
    text = re.findall(r'\b\w+\b', text)

    y = []

    # Remove special characters (keep only alphanumeric)
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords
    for i in text:
        if i not in ENGLISH_STOPWORDS:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# FASTAPI APP


app = FastAPI(
    title="Spam SMS Classifier",
    description="Machine Learning based Spam SMS Detection API",
    version="1.0.0"
)


# LOAD MODEL AND VECTORIZER


with open("model.pkl", "rb") as file:
    model = pickle.load(file)


with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


# HOME PAGE


@app.get("/", response_class=HTMLResponse)
def home():

    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    return html_content



# PREDICTION API


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
