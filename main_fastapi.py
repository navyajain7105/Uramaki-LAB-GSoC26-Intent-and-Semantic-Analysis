from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Import your functions from the other files
# (Assuming your NLP function from earlier is in a file called nlp_engine.py)
from nlp_engine import analyze_usability_text, discover_themes_with_bertopic 
from utils import extract_user_dialogue, generate_markdown_report

app = FastAPI(
    title="UX Intent Analysis API",
    description="An NLP-powered API for extracting actionable insights from UX transcripts.",
    version="1.0.0"
)

# Define the expected input schema
class TranscriptRequest(BaseModel):
    raw_transcript: str

# Define the expected output schema
class AnalysisResponse(BaseModel):
    parsed_user_quotes: int
    executive_report: str
    raw_analysis: List[Dict[str, Any]]

@app.post("/analyze-transcript", response_model=AnalysisResponse)
async def analyze_transcript_endpoint(request: TranscriptRequest):
    """
    End-to-end pipeline:
    1. Parses the transcript to isolate user dialogue.
    2. Runs the sarcasm-proof Zero-Shot NLP pipeline on each quote.
    3. Runs BERTopic for semantic clustering.
    4. Generates a Markdown summary report.
    """
    
    # Step 1: Parse out the moderator
    user_quotes = extract_user_dialogue(request.raw_transcript)
    
    # Step 2: Analyze Intent/Issue/Emotion
    results = []
    for quote in user_quotes:
        results.append(analyze_usability_text(quote))

    # Step 3: Semantic Clustering (BERTopic)
    themes = discover_themes_with_bertopic(user_quotes)

    # Step 4: Generate Executive Report
    markdown_report = generate_markdown_report(results, top_themes=themes)
    
    # Return the full package
    return {
        "parsed_user_quotes": len(user_quotes),
        "executive_report": markdown_report,
        "raw_analysis": results
    }
