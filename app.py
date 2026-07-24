import json
import sys
import os
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '.')
from src.prediction import predict

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="shield",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        padding: 2rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { color: white; font-size: 2.5rem; margin: 0; }
    .hero p  { color: #94a3b8; margin-top: 0.5rem; }

    .safe-box {
        background: #064e3b;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    .phish-box {
        background: #7f1d1d;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    .verdict { font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; }
    .warning-item {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.9rem;
    }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### PhishGuard")
    st.markdown("ML-Based Phishing Detector")
    st.divider()
    page = st.radio(
        "Navigate",
        ["URL Scanner", "Model Dashboard", "How It Works"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Random Forest | 90% Accuracy | 10,245 URLs trained")

# ── Header ────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>PhishGuard</h1>
    <p>Machine Learning-Based Phishing Website Detection</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# PAGE 1 — URL SCANNER
# ════════════════════════════════════════════════
if page == "URL Scanner":

    st.subheader("Enter a URL to analyze")

    # Initialize session state for URL
    if 'url_input' not in st.session_state:
        st.session_state.url_input = ""

    # Quick example buttons — they SET the session state
    st.markdown("**Quick examples:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Google (Safe)"):
        st.session_state.url_input = "https://www.google.com"
    if col2.button("Fake PayPal"):
        st.session_state.url_input = "http://paypal-secure-login.com/verify@fake.com"
    if col3.button("IP-based URL"):
        st.session_state.url_input = "http://192.168.1.1/login.php"

    # Text input reads FROM session state
    url_input = st.text_input(
        "URL",
        value=st.session_state.url_input,
        placeholder="https://www.example.com",
        label_visibility="collapsed"
    )

    if st.button("Analyze URL", type="primary") and url_input:

        with st.spinner("Analyzing..."):
            result = predict(url_input)

        st.divider()

        # ── Result columns ────────────────────────────
        left, right = st.columns(2)

        with left:
            is_phish  = result['label'] == 1
            box_class = "phish-box" if is_phish else "safe-box"
            icon      = "PHISHING DETECTED" if is_phish else "LEGITIMATE"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="verdict">{icon}</div>
                <div>Risk Level: {result['risk_level']}</div>
                <div style="margin-top:0.5rem; font-size:0.9rem; opacity:0.8;">
                    {result['confidence']}% phishing probability
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1rem; padding:0.8rem;
                        background:#1e293b; border-radius:8px;
                        font-family:monospace; font-size:0.85rem;
                        color:#94a3b8; word-break:break-all;">
                {result['url']}
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("**Phishing Probability**")
            st.progress(result['confidence'] / 100)
            st.metric("Phishing",   f"{result['confidence']}%")
            st.metric("Legitimate", f"{100 - result['confidence']}%")

        st.divider()

        # ── Features + Warnings ───────────────────────
        feat_col, warn_col = st.columns(2)

        with feat_col:
            st.markdown("**URL Feature Breakdown**")
            features = result['features']
            display = {
                'URL Length':       features['length_url'],
                'Domain Length':    features['length_hostname'],
                'Has IP Address':   'Yes' if features['ip'] else 'No',
                'Dot Count':        features['nb_dots'],
                'Hyphen Count':     features['nb_hyphens'],
                'Has @ Symbol':     'Yes' if features['nb_at'] else 'No',
                'Slash Count':      features['nb_slash'],
                'Subdomain Count':  features['nb_subdomains'],
                'HTTPS':            'Yes' if features['https_token'] else 'No',
                'Phish Keywords':   features['phish_hints'],
                'Prefix/Suffix(-)': 'Yes' if features['prefix_suffix'] else 'No',
            }
            for name, val in display.items():
                st.markdown(
                    f'<div class="feature-row">'
                    f'<span style="color:#94a3b8">{name}</span>'
                    f'<span style="color:#60a5fa; font-weight:600">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with warn_col:
            st.markdown("**Security Warnings**")
            for w in result['warnings']:
                st.markdown(
                    f'<div class="warning-item">! {w}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("**What should you do?**")
            if is_phish:
                st.error(
                    "Do NOT visit this website. "
                    "It shows strong phishing indicators. "
                    "Do not enter any personal information."
                )
            else:
                st.success(
                    "This URL appears safe. "
                    "Always verify the domain spelling "
                    "and look for HTTPS before entering data."
                )
                

# ════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ════════════════════════════════════════════════
elif page == "Model Dashboard":

    st.subheader("Model Performance Dashboard")

    # Load results
    try:
        with open("models/evaluation_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("Run model_training.py first to generate results.")
        st.stop()

    # Best model metrics
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    best      = results[best_name]

    st.markdown(f"**Best Model: {best_name}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{best['accuracy']}%")
    m2.metric("Precision", f"{best['precision']}%")
    m3.metric("Recall",    f"{best['recall']}%")
    m4.metric("F1-Score",  f"{best['f1_score']}%")

    st.divider()

    # Comparison table
    st.markdown("**All Models Comparison**")
    rows = []
    for name, m in results.items():
        rows.append({
            'Model':     name + (' (Best)' if name == best_name else ''),
            'Accuracy':  f"{m['accuracy']}%",
            'Precision': f"{m['precision']}%",
            'Recall':    f"{m['recall']}%",
            'F1-Score':  f"{m['f1_score']}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # Charts
    charts = {
        'Model Comparison':   'visualizations/model_comparison.png',
        'Feature Importance': 'visualizations/feature_importance.png',
        'Class Distribution': 'visualizations/class_distribution.png',
    }
    for title, path in charts.items():
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)
        else:
            st.warning(f"{title} chart not found. Run visualizations.py first.")

    # Confusion matrices
    st.markdown("**Confusion Matrices**")
    cols = st.columns(2)
    for i, name in enumerate(results.keys()):
        safe = name.replace(' ', '_').lower()
        path = f"visualizations/cm_{safe}.png"
        if os.path.exists(path):
            cols[i % 2].image(path, caption=name, use_column_width=True)


# ════════════════════════════════════════════════
# PAGE 3 — HOW IT WORKS
# ════════════════════════════════════════════════
elif page == "How It Works":

    st.subheader("How PhishGuard Works")

    tab1, tab2, tab3 = st.tabs(["Overview", "Features", "Models"])

    with tab1:
        st.markdown("""
        ## What is Phishing?
        Phishing is a cyberattack where criminals create fake websites
        that look identical to real ones to steal your credentials.

        ## Our Approach
        Instead of accessing the website (which is dangerous),
        we analyze only the **URL structure** — the web address itself.
        Phishing URLs have measurable patterns our ML model learns to detect.

        ## Pipeline
        User enters URL
            |
        Extract 15 features from URL string
            |
        Scale features with StandardScaler
            |
        Random Forest predicts: Phishing or Legitimate
            |
        Show result + confidence + warnings
        """)

    with tab2:
        st.markdown("## The 15 Features We Analyze")
        feature_info = [
            ("length_url",       "Total URL length - phishing URLs tend to be longer"),
            ("length_hostname",  "Domain name length - fake domains are longer"),
            ("ip",               "Uses raw IP instead of domain - strong signal"),
            ("nb_dots",          "Number of dots - more dots = subdomain stacking"),
            ("nb_hyphens",       "Number of hyphens - used to fake real domains"),
            ("nb_at",            "Has @ symbol - browser ignores content before it"),
            ("nb_slash",         "Slash count - deep paths hide malicious routes"),
            ("nb_percent",       "Hex encoding - hides malicious characters"),
            ("nb_subdomains",    "Subdomain count - many subdomains = suspicious"),
            ("https_token",      "Has HTTPS - absence is a warning sign"),
            ("phish_hints",      "Suspicious keywords: login, verify, secure..."),
            ("domain_age",       "Domain age - phishing domains are newer"),
            ("page_rank",        "Google trust score - phishing sites rank low"),
            ("nb_redirection",   "Redirect count - multiple redirects hide destination"),
            ("prefix_suffix",    "Hyphen in domain - paypal-secure.com pattern"),
        ]
        for feat, desc in feature_info:
            with st.expander(feat):
                st.write(desc)

    with tab3:
        st.markdown("""
        ## Models Compared

        | Model | Strength | Weakness |
        |-------|----------|----------|
        | Random Forest | High accuracy, handles non-linear patterns | Slower |
        | Decision Tree | Easy to interpret | Overfits easily |
        | Logistic Regression | Fast, simple baseline | Linear only |
        | SVM | Strong on complex boundaries | Slow on large data |

        ## Why Random Forest Won
        It builds 100 decision trees and takes a majority vote.
        No single tree's mistake dominates - errors cancel out.
        This is called **ensemble learning**.
        """)