import unittest
from app.preprocessing.text_preprocessing import preprocess_text

class TestPreprocessing(unittest.TestCase):

    def test_clean_text(self):
        result = preprocess_text("BANK!!! 😊")
        self.assertIsInstance(result, str)

if __name__ == "__main__":
    unittest.main()