# Spam SMS Classifier

A Machine Learning based Spam SMS Detection API built with FastAPI and scikit-learn. Paste any SMS message and instantly find out whether it's spam or not, along with confidence scores.

## Features

- Real-time SMS spam classification
- Confidence scores for both spam and not-spam predictions
- Clean, simple web interface
- REST API (`/predict`) for integration into other apps
- Interactive API docs via Swagger UI (`/docs`)

## Tech Stack

- **Backend:** FastAPI
- **Server:** Uvicorn
- **ML:** scikit-learn (TF-IDF + classifier), NLTK (text preprocessing)
- **Frontend:** Plain HTML/CSS/JS

## Project Structure

```
.
├── app.py              # FastAPI application
├── index.html          # Web UI
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies (pinned versions)
├── runtime.txt          # Python version for deployment
├── render.yaml          # Render deployment config
└── .gitignore
```

## Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Create virtual environment (Python 3.12 recommended)
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run the app
python app.py
```

Visit `http://127.0.0.1:8000` in your browser.

## API Usage

**Endpoint:** `POST /predict`

**Form field:** `message` (string)

**Example Response:**
```json
{
  "message": "Congratulations! You won a free lottery, claim now",
  "prediction": "Spam",
  "spam_probability": 98.5,
  "not_spam_probability": 1.5
}
```

## Deployment

This project is configured for one-click deployment on [Render](https://render.com) using `render.yaml`.

1. Push this repository to GitHub.
2. Create a new Web Service on Render and connect this repo.
3. Render will auto-detect `render.yaml` and handle the build/start commands.
4. Once deployed, your app will be live at a public `.onrender.com` URL.

## License

This project is open for personal and educational use.
