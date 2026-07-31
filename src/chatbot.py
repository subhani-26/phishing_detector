# src/chatbot.py
# AI Chatbot using Groq API - Completely Free
# Explains phishing detection results intelligently

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)


# ── System Prompt ─────────────────────────────
SYSTEM_PROMPT = """You are CyberGuard, an AI 
cybersecurity assistant integrated into PhishGuard 
— a machine learning based phishing detection system.

Your job is to:
1. Provide a brief description and the likely category of the website based on the URL.
2. Explain in simple language why a URL was flagged 
   as phishing or legitimate
3. Handle false positives intelligently — if a 
   well known site like GeeksForGeeks, Wikipedia, 
   YouTube is flagged, explain it is likely safe 
   but has features that triggered the detector
4. If a brand new website is flagged, explain that 
   new sites naturally have low page rank and domain 
   age which can trigger false positives
5. Give practical safety advice the user can act on
6. Answer any follow up questions about phishing, 
   cybersecurity, or URL safety
7. Never be overly technical — explain like talking 
   to a non-technical person

Always be helpful, honest, and clear.
Keep responses under 250 words.
Never say you cannot help."""


def get_chatbot_response(url: str, 
                         result: dict,
                         user_question: str = None,
                         chat_history: list = None) -> str:
    """
    Gets AI explanation from Groq for a URL analysis.
    
    Parameters:
        url: The analyzed URL string
        result: Full prediction result from predict()
        user_question: Optional follow up question
        chat_history: Previous messages in conversation
    
    Returns:
        str: AI generated explanation
    """
    
    # Extract result details
    features   = result['features']
    warnings   = result['warnings']
    verdict    = result['prediction']
    confidence = result['confidence']
    risk       = result['risk_level']
    
    # Build URL context for AI
    url_context = f"""
URL Analyzed: {url}
ML Model Verdict: {verdict}
Phishing Confidence: {confidence}%
Risk Level: {risk}

URL Features Detected:
- URL Length: {features['length_url']} characters
- Has IP Address: {'Yes - SUSPICIOUS' if features['ip'] else 'No - Normal'}
- Number of Dots: {features['nb_dots']}
- Has @ Symbol: {'Yes - SUSPICIOUS' if features['nb_at'] else 'No - Normal'}
- Has HTTPS: {'Yes - Good' if features['https_token'] else 'No - Suspicious'}
- Subdomain Count: {features['nb_subdomains']}
- Suspicious Keywords: {features['phish_hints']} found
- Hyphen in Domain: {'Yes - Suspicious' if features['prefix_suffix'] else 'No - Normal'}
- Slash Count: {features['nb_slash']}

Security Warnings Generated:
{chr(10).join(f'- {w}' for w in warnings)}
"""

    # Build messages list
    messages = []
    
    # Add chat history if exists
    if chat_history:
        for msg in chat_history:
            messages.append({
                "role":    msg['role'],
                "content": msg['content']
            })
    
    # Build user message
    if user_question:
        # Follow up question — include context
        user_message = f"""
{url_context}

My question: {user_question}
"""
    else:
        # First analysis — ask for explanation
        user_message = f"""
Please analyze this URL detection result and explain 
it clearly to me:

{url_context}

Tell me:
1. What is the likely description and category of this website based on the URL?
2. Why was this verdict given?
3. Which features are most concerning or reassuring?
4. What should I do — is it safe to visit?
5. Could this be a false positive?
"""
    
    messages.append({
        "role":    "user",
        "content": user_message
    })
    
    # Call Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Free, fast model
        messages=[
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            }
        ] + messages,
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content


def explain_feature(feature_name: str) -> str:
    """
    Explains what a URL feature means in simple terms.
    Uses AI for intelligent explanation.
    
    Parameters:
        feature_name: Name of the feature to explain
    
    Returns:
        str: Simple explanation of the feature
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""Explain what the URL 
feature '{feature_name}' means in phishing detection.
Keep it under 3 sentences.
Use simple language a student can understand.
Give one real example of how this feature appears 
in a phishing URL."""
            }
        ],
        max_tokens=150,
        temperature=0.5
    )
    
    return response.choices[0].message.content


def get_safety_advice(verdict: str, 
                      confidence: float) -> str:
    """
    Gives specific safety advice based on verdict.
    
    Parameters:
        verdict: 'Phishing' or 'Legitimate'
        confidence: Phishing probability percentage
    
    Returns:
        str: Safety advice
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""A URL was classified as 
{verdict} with {confidence}% phishing confidence.

Give me 3 specific safety tips about what to do next.
Be practical and actionable.
Keep under 100 words."""
            }
        ],
        max_tokens=150,
        temperature=0.5
    )
    
    return response.choices[0].message.content


# ── Quick Test ────────────────────────────────
if __name__ == "__main__":
    
    # Test with a fake result
    test_result = {
        'prediction': 'Phishing',
        'confidence': 96.73,
        'risk_level': 'CRITICAL',
        'features': {
            'length_url':      45,
            'ip':              0,
            'nb_dots':         3,
            'nb_at':           1,
            'https_token':     0,
            'nb_subdomains':   2,
            'phish_hints':     4,
            'prefix_suffix':   1,
            'nb_slash':        4,
            'nb_hyphens':      2,
            'nb_percent':      0,
            'length_hostname': 25,
            'nb_redirection':  1,
            'domain_age':      1000,
            'page_rank':       2
        },
        'warnings': [
            "Contains @ symbol",
            "Contains 4 suspicious keywords",
            "Hyphen found in domain",
            "No HTTPS detected"
        ]
    }
    
    test_url = "http://paypal-secure-login.com/verify@fake.com"
    
    print("Testing Groq Chatbot...")
    print("=" * 50)
    
    response = get_chatbot_response(test_url, test_result)
    print("AI Response:")
    print(response)