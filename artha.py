"""
Artha - AI Financial Advisor Module
Provides AI-powered financial advice using OpenAI.
"""

import os

try:
    import openai
except ImportError:
    openai = None

def generate_ai_advice(recommendation, confidence, alerts, news_sentiment, company_name):
    """
    Generates AI-powered financial advice using OpenAI.
    """
    if not openai:
        return "Artha: AI advice unavailable. OpenAI library not installed."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Artha: AI advice unavailable. OpenAI API key not set in environment variables."

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = f"You are Artha, a wise financial advisor AI. Based on the following stock analysis for {company_name}, provide concise, personalized financial advice. Include potential risks, considerations, and next steps. Keep it under 200 words.\n\nSwing Trader Recommendation: {recommendation}\nConfidence: {confidence}%\nAlerts: {', '.join(alerts)}\nOverall News Sentiment: {news_sentiment}\n\nArtha's Advice:"

    try:
        response = client.chat.completions.create(
            model="qwen/qwen2.5-vl-72b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
        return f"Artha: {advice}"
    except Exception as e:
        return f"Artha: Sorry, I couldn't generate advice right now. Error: {str(e)}"