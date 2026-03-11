# Helper functions for saving inference results to CSV file

import csv
import os
from datetime import datetime

CSV_FILE = "inference_results.csv"

def save_to_csv(text, model_type, label, score):
    """
    Append a new inference result to CSV.
    
    :param text: Input text
    :param model_type: 'finbert' or 'absa'
    :param label: Predicted label
    :param score: Confidence score
    """
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "text", "model_type", "label", "score"])
        writer.writerow([datetime.now(), text, model_type, label, score])