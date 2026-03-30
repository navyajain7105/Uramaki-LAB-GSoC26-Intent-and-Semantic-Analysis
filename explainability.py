import shap
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("Loading PyTorch Models for SHAP Explainability...")

# 1. Load the raw model and tokenizer (Bypassing the pipeline for deep access)
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def generate_shap_explanation(quote: str, target_label: str, hypothesis_type: str = "intent"):
    """
    A custom wrapper to make SHAP work with Zero-Shot NLI models.
    """
    # Define the hypothesis based on what we are testing
    if hypothesis_type == "intent":
        hypothesis = f"Regardless of sarcasm, the user's actual goal is to {target_label}."
    elif hypothesis_type == "issue":
        hypothesis = f"The actual technical problem the user is experiencing is {target_label}."
    else:
        hypothesis = f"Despite their tone, the user's true underlying feeling is {target_label}."

    # 2. The Custom Prediction Function for SHAP
    def custom_nli_predict(texts):
        """
        SHAP will pass hundreds of mutated versions of the quote here.
        We pair each one with the hypothesis and return the 'entailment' probability.
        """
        # Pair every mutated text with our fixed hypothesis
        pairs = [[text, hypothesis] for text in texts]
        
        # Tokenize
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        
        # Run through the model
        with torch.no_grad():
            outputs = model(**inputs)
            
            # BART MNLI logits order: [contradiction, neutral, entailment]
            # We want the probability of entailment (index 2)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            entailment_probs = probs[:, 2].numpy()
            
        return entailment_probs

    # 3. Initialize the SHAP Explainer
    print(f"\nRunning SHAP Analysis for '{target_label}'...")
    explainer = shap.Explainer(custom_nli_predict, tokenizer)
    
    # 4. Calculate SHAP values for the specific quote
    shap_values = explainer([quote])
    
    return shap_values

# --- Run the Test ---
if __name__ == "__main__":
    # The tricky sarcastic quote
    test_quote = "I love clicking buttons that don't respond."
    
    # Let's explain why the model thinks this is a "Report Bug" intent
    target = "Report Bug"
    
    shap_vals = generate_shap_explanation(test_quote, target, hypothesis_type="intent")
    
    # If running in a Jupyter/Colab notebook, this next line renders the visual:
    print(shap.plots.text(shap_vals))
    print("\nSHAP values calculated successfully!")
