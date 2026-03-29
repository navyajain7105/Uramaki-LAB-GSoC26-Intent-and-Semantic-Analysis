from transformers import pipeline
from bertopic import BERTopic
import pandas as pd

print("Loading AI Models... (This takes a moment)")

# Initialize the powerhouse Zero-Shot model
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the standard UX Taxonomy
INTENT_LABELS = ["Find Information", "Complete Task", "Request Feature", "Report Bug", "Provide Feedback"]
ISSUE_LABELS = ["Confusing Navigation", "Broken Feature", "Slow Performance", "Missing Information", "No Issue"]
EMOTION_LABELS = ["Frustrated", "Satisfied", "Confused", "Angry", "Neutral"]

def analyze_usability_text(quote: str) -> dict:
    """
    A robust zero-shot pipeline designed to bypass sarcasm and extract true UX intent.
    Uses highly specific hypothesis templates to force logical inference over simple keyword matching.
    """
    
    # Predict Intent
    intent_result = zero_shot_classifier(
        quote, 
        candidate_labels=INTENT_LABELS,
        multi_label=True,
        hypothesis_template="Regardless of sarcasm, the user's actual goal is to {}."
    )
    top_intent = intent_result['labels'][0]
    intent_conf = round(intent_result['scores'][0], 3)
    
    # Predict Perceived Issue
    issue_result = zero_shot_classifier(
        quote, 
        candidate_labels=ISSUE_LABELS,
        multi_label=True,
        hypothesis_template="The actual technical problem the user is experiencing is {}."
    )
    top_issue = issue_result['labels'][0]
    issue_conf = round(issue_result['scores'][0], 3)
    
    # Predict True Emotion
    emotion_result = zero_shot_classifier(
        quote, 
        candidate_labels=EMOTION_LABELS,
        multi_label=False, # We only want the single strongest true emotion
        hypothesis_template="Despite their tone, the user's true underlying feeling is {}."
    )
    top_emotion = emotion_result['labels'][0]
    emotion_conf = round(emotion_result['scores'][0], 3)
    
    return {
        "raw_quote": quote,
        "analysis": {
            "primary_intent": top_intent,
            "perceived_issue": top_issue,
            "true_emotion": top_emotion
        },
        "explainability_metrics": {
            "intent_confidence": intent_conf,
            "issue_confidence": issue_conf,
            "emotion_confidence": emotion_conf
        }
    }

def discover_themes_with_bertopic(quotes: list) -> str:
    """
    Uses BERTopic to perform unsupervised semantic clustering on a list of quotes.
    Returns a formatted string of the top discovered themes.
    """
    if len(quotes) < 5:
        return "Not enough data for semantic clustering (requires 5+ quotes)."
        
    try:
        # min_topic_size is kept small for PoC purposes
        topic_model = BERTopic(language="english", calculate_probabilities=False, min_topic_size=2)
        topics, _ = topic_model.fit_transform(quotes)
        
        topic_info = topic_model.get_topic_info()
        # Return the top themes (excluding outliers labeled as -1)
        clean_topics = topic_info[topic_info['Topic'] != -1]
        
        if clean_topics.empty:
            return "No distinct clusters found."
            
        return clean_topics[['Count', 'Name']].head(3).to_string(index=False)
    except Exception as e:
        return f"Clustering skipped or failed: {str(e)}"