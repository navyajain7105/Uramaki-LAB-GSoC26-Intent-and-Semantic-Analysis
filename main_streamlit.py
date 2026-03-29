import streamlit as st
import pandas as pd
from nlp_engine import analyze_usability_text, discover_themes_with_bertopic
from utils import extract_user_dialogue

# Set up the page
st.set_page_config(page_title="RUXAILAB UX Analyzer", layout="wide")
st.title("🎯 UX Intent & Semantic Analyzer")
st.markdown("Transform raw usability testing transcripts into actionable, sarcasm-proof insights.")

# Input Section
st.sidebar.header("Data Input")
transcript_input = st.sidebar.text_area(
    "Paste Usability Transcript Here:",
    value="[Moderator]: Did you find the checkout easy?\n[User]: I love clicking buttons that don't respond. Took 5 minutes.\n[Moderator]: Anything else?\n[User]: Navigation feels like a maze.",
    height=300
)

analyze_button = st.sidebar.button("Analyze Transcript")

if analyze_button and transcript_input:
    with st.spinner("Parsing dialogue and running Zero-Shot NLP..."):
        # 1. Parse Dialogue
        user_quotes = extract_user_dialogue(transcript_input)
        
        if not user_quotes:
            st.warning("No user dialogue found. Make sure to use [User]: tags.")
        else:
            # 2. Run Analysis
            results = [analyze_usability_text(q) for q in user_quotes]
            
            # Create a clean dataframe for the UI
            df = pd.DataFrame([{
                "Raw Quote": r["raw_quote"],
                "Intent": r["analysis"]["primary_intent"],
                "Issue": r["analysis"]["perceived_issue"],
                "True Emotion": r["analysis"]["true_emotion"]
            } for r in results])
            
            # 3. Top-Level Metrics
            st.header("Executive Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Utterances Analyzed", len(user_quotes))
            col2.metric("Top Issue", df["Issue"].mode()[0] if not df.empty else "N/A")
            col3.metric("Dominant Emotion", df["True Emotion"].mode()[0] if not df.empty else "N/A")
            
            # 4. Detailed Breakdown
            st.header("Utterance Breakdown")
            st.dataframe(df, use_container_width=True)
            
            # 5. Semantic Themes (BERTopic)
            st.header("Semantic Clusters Discovered")
            if len(user_quotes) >= 5: # Need a minimum amount of data for clustering
                themes = discover_themes_with_bertopic(user_quotes)
                st.text(themes)
            else:
                st.info("Provide at least 5 user utterances to activate BERTopic semantic clustering.")