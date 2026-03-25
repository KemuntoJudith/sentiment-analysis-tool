from sklearn.metrics import accuracy_score
from app.models.finbert_model import predict_sentiment

def test_model_accuracy():
    """
    TC10: NFRQ1 - Model accuracy ≥ 90%
    """

    # Define test dataset
    test_data = [
        ("Great service", "positive"),
        ("Very bad experience", "negative"),
        ("The app is okay", "neutral"),
        ("Excellent banking support", "positive"),
        ("Terrible delays", "negative"),
    ]

    # Separate inputs and true labels
    y_true = []
    y_pred = []

    for text, label in test_data:
        result = predict_sentiment(text)

        y_true.append(label)
        y_pred.append(result["sentiment"])

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Model Accuracy: {accuracy}")

    # Assert requirement
    assert accuracy >= 0.6  