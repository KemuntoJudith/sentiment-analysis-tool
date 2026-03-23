import unittest
from app.models.finbert_model import predict_sentiment

class TestModel(unittest.TestCase):

    def test_prediction(self):
        result = predict_sentiment("Great service")
        self.assertIn(result["sentiment"], ["positive","neutral","negative"])

if __name__ == "__main__":
    unittest.main()