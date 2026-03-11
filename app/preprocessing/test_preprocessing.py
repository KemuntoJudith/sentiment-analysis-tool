# Test the preprocessing module

from text_preprocessing import preprocess_text

text = "The mobile banking app is VERY slow!!! Visit http://bank.com"

cleaned = preprocess_text(text)

print("Original:", text)
print("Processed:", cleaned)