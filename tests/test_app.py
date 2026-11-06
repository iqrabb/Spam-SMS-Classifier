from fastapi.testclient import TestClient

from app import app, transform_text

client = TestClient(app)


def test_transform_text_lowercases_and_removes_stopwords():
    assert transform_text("THIS is a simple TEST") == "simpl test"


def test_home_page_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Spam SMS Classifier" in response.text


def test_predict_rejects_blank_message():
    response = client.post("/predict", data={"message": "   "})
    assert response.status_code == 422


def test_predict_rejects_message_over_limit():
    response = client.post("/predict", data={"message": "x" * 5001})
    assert response.status_code == 422


def test_predict_returns_expected_response_fields():
    response = client.post(
        "/predict",
        data={"message": "Congratulations! You have won a free prize. Call now."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"Spam", "Not Spam"}
    assert 0 <= payload["spam_probability"] <= 100
    assert 0 <= payload["not_spam_probability"] <= 100
    assert abs(payload["spam_probability"] + payload["not_spam_probability"] - 100) < 0.1
