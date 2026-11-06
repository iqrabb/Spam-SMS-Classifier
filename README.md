# Spam SMS Classifier

A web-based machine-learning project that classifies an SMS message as **Spam** or **Not Spam**. It combines a scikit-learn text-classification pipeline with a FastAPI backend and a lightweight HTML/CSS/JavaScript interface.

> **Project type:** Educational machine-learning application. Predictions are model estimates, not a guarantee that a message is safe.

## What it does

- Accepts an SMS message through a browser interface or API request.
- Applies text preprocessing: lowercasing, regex tokenization, stopword removal, and Porter stemming.
- Converts the processed text with a saved TF-IDF vectorizer.
- Uses a saved scikit-learn model to predict the class.
- Returns spam and not-spam probability estimates.
- Exposes interactive API documentation through Swagger UI.

## Technology

| Area | Tools |
|---|---|
| API | FastAPI, Uvicorn |
| Machine learning | scikit-learn |
| Text features | TF-IDF |
| Text preprocessing | Python regex, NLTK PorterStemmer |
| Frontend | HTML, CSS, JavaScript |
| Saved artifacts | `model.pkl`, `vectorizer.pkl` |

## How a prediction works

1. A user submits an SMS message in the web interface or sends it to `POST /predict`.
2. FastAPI passes the message to the preprocessing function.
3. The TF-IDF vectorizer transforms the processed text into numerical features.
4. The trained model predicts the class and calculates class probabilities.
5. The API returns the original message, predicted label, and both probability values.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/iqrabb/Spam-SMS-Classifier.git
cd Spam-SMS-Classifier
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open these local addresses:

- Web interface: http://127.0.0.1:8000/
- Swagger API documentation: http://127.0.0.1:8000/docs

The application expects `model.pkl` and `vectorizer.pkl` in the project root when it starts.

## API reference

### `POST /predict`

Accepts form data (not JSON).

| Field | Type | Required |
|---|---|---|
| `message` | string | Yes |

Example request with cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "message=Congratulations, you have won a prize"
```

Example response shape (values are illustrative):

```json
{
  "message": "Congratulations, you have won a prize",
  "prediction": "Spam",
  "spam_probability": 98.5,
  "not_spam_probability": 1.5
}
```

## Repository contents

- `app.py` — FastAPI application, preprocessing, and prediction endpoint.
- `index.html` — browser interface.
- `model.pkl` — serialized trained model.
- `vectorizer.pkl` — serialized TF-IDF vectorizer.
- `requirements.txt` — Python dependencies.

## Current scope and limitations

- The repository contains the application and serialized model artifacts; the dataset and model-training notebook/script are not currently included. This means the training process cannot yet be reproduced from this repository alone.
- No independently verified test metrics are documented here. Performance should be evaluated on a held-out dataset before making claims about accuracy or production readiness.
- Probability values are model outputs and should not be interpreted as calibrated certainty.
- The app binds to `127.0.0.1` in the supplied run configuration, so it is configured for local use by default.
- Only load pickle files from sources you trust. Unpickling an untrusted file can execute arbitrary code.

## Potential next improvements

- Add the dataset source and a reproducible training/evaluation script.
- Document the dataset, split strategy, and precision/recall/F1 results.
- Add automated tests for preprocessing and API behavior.
- Add input validation, logging, and deployment configuration.
- Include screenshots or a short demo of the interface.

---

Built as a practical demonstration of text preprocessing, TF-IDF feature extraction, classification, and API integration.
