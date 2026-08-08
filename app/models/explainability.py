# Functionality for explaining model predictions using LIME (Local Interpretable Model-agnostic Explanations)

from lime.lime_text import LimeTextExplainer
from app.models.finbert_model import predict_proba

CLASS_NAMES = ["negative", "neutral", "positive"]

explainer = LimeTextExplainer(class_names=CLASS_NAMES)

def explain_with_lime(text):

    # Prevent invalid inputs
    if not text or len(text.strip().split()) < 3:
        return None

    try:
        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=10,
            num_samples=100 
        )

        return explanation 

    except Exception as e:
        print(f"LIME error: {e}")
        return None