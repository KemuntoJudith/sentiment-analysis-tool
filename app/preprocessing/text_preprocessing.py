# Clean and normalize input text before it is sent to the sentiment model

import re

def preprocess_text(text: str) -> str:

    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = text.strip()

    return text