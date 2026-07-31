# app.py — PhishGuard Professional UI (Final Version)

import json
import sys
import os
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import pandas as pd

sys.path.insert(0, '.')
from src.prediction import predict
from groq import Groq
from src.chatbot import (
    get_chatbot_response,
    explain_feature,
    get_safety_advice,
    SYSTEM_PROMPT,
    GROQ_API_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"   
)




# ── Professional Clean CSS ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e9ecef;
    
}
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}


/* ── Header ── */
.pg-header {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pg-header-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin: 0; }
.pg-header-sub   { font-size: 0.8rem; color: #6c757d; margin: 0; }
.pg-badge        { font-size: 0.72rem; padding: 3px 10px; border-radius: 20px; font-weight: 500; margin-left: auto; }
.pg-badge-blue   { background: #e8f0fe; color: #1a73e8; }

/* ── Cards ── */
.pg-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.pg-card-title {
    font-size: 0.8rem; font-weight: 500; color: #6c757d;
    margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── Result boxes ── */
.pg-result-phishing { background: #fff5f5; border: 1px solid #fc8181; border-radius: 12px; padding: 1.25rem; text-align: center; }
.pg-result-safe     { background: #f0fff4; border: 1px solid #68d391; border-radius: 12px; padding: 1.25rem; text-align: center; }
.pg-verdict         { font-size: 1.3rem; font-weight: 600; margin: 0.4rem 0; }
.pg-verdict-phishing { color: #c53030; }
.pg-verdict-safe     { color: #276749; }
.pg-risk-badge       { display: inline-block; font-size: 0.72rem; font-weight: 500; padding: 3px 10px; border-radius: 20px; margin-top: 4px; }
.pg-risk-critical { background: #fed7d7; color: #c53030; }
.pg-risk-high     { background: #feebc8; color: #c05621; }
.pg-risk-medium   { background: #fefcbf; color: #b7791f; }
.pg-risk-low      { background: #c6f6d5; color: #276749; }

/* ── Feature rows ── */
.pg-feature-row      { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f3f4; font-size: 0.82rem; }
.pg-feature-label    { color: #6c757d; }
.pg-feature-val-safe    { color: #276749; font-weight: 500; }
.pg-feature-val-danger  { color: #c53030; font-weight: 500; }
.pg-feature-val-neutral { color: #1a1a2e; font-weight: 500; }

/* ── Warnings ── */
.pg-warning        { display: flex; align-items: flex-start; gap: 8px; padding: 7px 10px; border-radius: 8px; margin-bottom: 6px; font-size: 0.82rem; }
.pg-warning-danger { background: #fff5f5; color: #c53030; }
.pg-warning-medium { background: #fffaf0; color: #c05621; }
.pg-warning-safe   { background: #f0fff4; color: #276749; }

/* ── Metric cards ── */
.pg-metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1rem; }
.pg-metric       { background: #ffffff; border: 1px solid #e9ecef; border-radius: 10px; padding: 1rem; text-align: center; }
.pg-metric-value { font-size: 1.5rem; font-weight: 600; color: #1a73e8; margin: 0; }
.pg-metric-label { font-size: 0.75rem; color: #6c757d; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.04em; }

/* ── Progress bar ── */
.pg-bar-wrap      { background: #e9ecef; border-radius: 99px; height: 6px; overflow: hidden; margin: 6px 0; }
.pg-bar-fill-blue  { background: #1a73e8; height: 100%; border-radius: 99px; }
.pg-bar-fill-green { background: #38a169; height: 100%; border-radius: 99px; }

/* ── Chat bubbles ── */
.pg-bubble-bot  { background: #ffffff; border: 1px solid #e9ecef; border-radius: 12px; padding: 10px 14px; font-size: 0.85rem; color: #1a1a2e; line-height: 1.6; margin-bottom: 8px; }
.pg-bubble-user { background: #e8f0fe; border-radius: 12px; padding: 10px 14px; font-size: 0.85rem; color: #1a1a2e; margin-bottom: 8px; text-align: right; }

/* ── Chat panel inside page ── */
.pg-chat-panel {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1rem;
}
.pg-chat-header {
    background: #1a73e8;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.pg-chat-body { padding: 12px 14px; }
.chat-msg-bot  { background: #f8f9fa; border-radius: 10px; padding: 8px 12px; font-size: 0.82rem; color: #1a1a2e; margin-bottom: 8px; line-height: 1.5; }
.chat-msg-user { background: #e8f0fe; border-radius: 10px; padding: 8px 12px; font-size: 0.82rem; color: #1a1a2e; margin-bottom: 8px; text-align: right; line-height: 1.5; }

/* ── AI toggle button inside page (top right of AI section) ── */
.pg-ai-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

/* ── Pipeline ── */
.pg-pipeline  { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.pg-step      { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 6px 12px; font-size: 0.78rem; color: #1a1a2e; font-weight: 500; }
.pg-step-active { background: #e8f0fe; border: 1px solid #a8c7fa; color: #1a73e8; }
.pg-arrow     { color: #adb5bd; font-size: 0.85rem; }

/* ── How it works ── */
.pg-info-card  { background: #ffffff; border: 1px solid #e9ecef; border-radius: 12px; padding: 1rem 1.25rem; height: 100%; }
.pg-info-title { font-size: 0.9rem; font-weight: 600; color: #1a1a2e; margin: 8px 0 6px 0; }
.pg-info-body  { font-size: 0.82rem; color: #6c757d; line-height: 1.6; margin: 0; }

/* ── URL display ── */
.pg-url-display { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 8px 12px; font-family: monospace; font-size: 0.78rem; color: #6c757d; word-break: break-all; margin-top: 10px; }

/* ── Section label ── */
.pg-section-label { font-size: 0.75rem; font-weight: 500; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 10px 0; }

/* ── Hide Streamlit chrome ── */
/* ── Hide Streamlit chrome ── */
#MainMenu                  { visibility: hidden; }
footer                     { visibility: hidden; }
header                     { visibility: hidden; }

/* BUT keep the sidebar toggle button visible */
header [data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display:    flex    !important;
    opacity:    1       !important;
}
/* ── Buttons ── */
.stButton > button {
    border-radius: 8px; font-family: 'Inter', sans-serif;
    font-size: 0.85rem; font-weight: 500;
    border: 1px solid #e9ecef; background: #ffffff;
    color: #1a1a2e; padding: 6px 14px; transition: all 0.15s;
}
.stButton > button:hover { border-color: #1a73e8; color: #1a73e8; background: #e8f0fe; }
[data-testid="baseButton-primary"] > button,
.stButton > button[kind="primary"] { background: #1a73e8; color: #ffffff; border-color: #1a73e8; }
[data-testid="baseButton-primary"] > button:hover,
.stButton > button[kind="primary"]:hover { background: #1557b0; border-color: #1557b0; color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <span style='font-size:1.3rem;'>🛡️</span>
            <div>
                <p style='font-size:0.95rem; font-weight:600; color:#1a1a2e; margin:0;'>PhishGuard</p>
                <p style='font-size:0.72rem; color:#6c757d; margin:0;'>ML Phishing Detector</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍  URL Scanner",
         "📊  Model Dashboard",
         "📚  How It Works"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style='font-size:0.75rem; color:#adb5bd; line-height:1.8; padding-bottom:0.5rem;'>
        <b style='color:#6c757d;'>Model</b><br>Random Forest<br>
        <b style='color:#6c757d;'>Accuracy</b><br>90.09%<br>
        <b style='color:#6c757d;'>Dataset</b><br>10,245 URLs<br>
        <b style='color:#6c757d;'>Features</b><br>15 URL features
    </div>
    """, unsafe_allow_html=True)


# ── Page Header ───────────────────────────────
page_titles = {
    "🔍  URL Scanner":     ("URL Scanner",    "Analyze any URL for phishing indicators"),
    "📊  Model Dashboard": ("Model Dashboard", "Performance metrics and visualizations"),
    "📚  How It Works":    ("How It Works",    "Understand the detection pipeline"),
}
title, subtitle = page_titles[page]

st.markdown(f"""
<div class="pg-header">
    <span style='font-size:1.2rem;'>🛡️</span>
    <div>
        <p class="pg-header-title">PhishGuard</p>
        <p class="pg-header-sub">{subtitle}</p>
    </div>
    <span class="pg-badge pg-badge-blue">{title}</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE 1 — URL SCANNER + AI (MERGED)
# ════════════════════════════════════════════
if page == "🔍  URL Scanner":

    # Session state
    for key, val in {
        'url_input': "", 'messages': [],
        'chat_result': None, 'chat_url': "",
        'show_chat': False, 'analyzed': False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Input Card ────────────────────────────
    st.markdown('<div class="pg-card">', unsafe_allow_html=True)
    st.markdown('<p class="pg-card-title">Enter a URL to analyze</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    if c1.button("✅  Google (safe)"):
        st.session_state.url_input = "https://www.google.com"
    if c2.button("🚨  Fake PayPal"):
        st.session_state.url_input = "http://paypal-secure-login.com/verify@fake.com"
    if c3.button("⚠️  IP-based URL"):
        st.session_state.url_input = "http://192.168.1.1/login.php"

    url_input = st.text_input(
        "URL", value=st.session_state.url_input,
        placeholder="https://www.example.com",
        label_visibility="collapsed"
    )
    analyze = st.button("Analyze URL", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Run Analysis ──────────────────────────
    if analyze and url_input:
        st.session_state.url_input  = url_input
        st.session_state.messages   = []
        st.session_state.show_chat  = False
        st.session_state.analyzed   = True

        with st.spinner("Analyzing URL..."):
            result = predict(url_input)
            st.session_state.chat_result = result
            st.session_state.chat_url    = url_input

        with st.spinner("PhishGuard AI is analyzing..."):
            try:
                explanation = get_chatbot_response(url_input, result)
                st.session_state.messages.append({"role": "assistant", "content": explanation})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Analysis complete. {str(e)}"})

    # ── Show Results ──────────────────────────
    if st.session_state.analyzed and st.session_state.chat_result:

        result   = st.session_state.chat_result
        is_phish = result['label'] == 1
        conf     = result['confidence']
        risk     = result['risk_level']
        features = result['features']

        risk_map = {
            "CRITICAL": "pg-risk-critical", "HIGH RISK": "pg-risk-high",
            "MEDIUM RISK": "pg-risk-medium", "LOW RISK": "pg-risk-low"
        }
        badge_cls   = risk_map.get(risk, "pg-risk-low")
        box_cls     = "pg-result-phishing" if is_phish else "pg-result-safe"
        verdict_cls = "pg-verdict-phishing" if is_phish else "pg-verdict-safe"
        icon        = "🚨" if is_phish else "✅"

        st.divider()

        # Verdict row
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"""
            <div class="{box_cls}">
                <div style='font-size:2rem;'>{icon}</div>
                <p class="pg-verdict {verdict_cls}">{result['prediction']}</p>
                <span class="pg-risk-badge {badge_cls}">{risk}</span>
                <div class="pg-url-display">{result['url']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="pg-card" style='margin-bottom:0;'>
                <p class="pg-card-title">Phishing probability</p>
                <p style='font-size:1.8rem; font-weight:600;
                           color:{"#c53030" if is_phish else "#276749"}; margin:0;'>{conf}%</p>
                <div class="pg-bar-wrap">
                    <div class="{"pg-bar-fill-blue" if is_phish else "pg-bar-fill-green"}"
                         style='width:{conf}%;'></div>
                </div>
                <div style='display:flex; justify-content:space-between;
                             font-size:0.75rem; color:#6c757d; margin-top:4px;'>
                    <span>Legitimate {100-conf:.2f}%</span>
                    <span>Phishing {conf}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── AI Explanation with toggle ────────
        ai_left, ai_right = st.columns([4, 1])
        with ai_left:
            st.markdown("""
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:10px;'>
                <span style='font-size:1.2rem;'>🤖</span>
                <p class="pg-section-label" style='margin:0;'>PhishGuard AI explanation</p>
                <span class="pg-badge pg-badge-blue" style='margin-left:0;'>Powered by Groq</span>
            </div>
            """, unsafe_allow_html=True)

        with ai_right:
            chat_label = "✕ Close chat" if st.session_state.show_chat else "💬 Ask AI"
            if st.button(chat_label, key="toggle_chat"):
                st.session_state.show_chat = not st.session_state.show_chat
                st.rerun()

        # Auto AI explanation
        if st.session_state.messages:
            st.markdown(
                f'<div class="pg-bubble-bot">{st.session_state.messages[0]["content"]}</div>',
                unsafe_allow_html=True
            )

        # ── Expandable Chat Panel ─────────────
        if st.session_state.show_chat:
            st.markdown("""
            <div class="pg-chat-panel">
                <div class="pg-chat-header">
                    <span style='font-size:1.2rem;'>🤖</span>
                    <p style='color:white; font-size:0.9rem; font-weight:600; margin:0;'>PhishGuard AI</p>
                    <span style='font-size:0.72rem; background:rgba(255,255,255,0.2);
                                  color:white; padding:2px 8px; border-radius:20px; margin-left:auto;'>
                        Ask anything
                    </span>
                </div>
                <div class="pg-chat-body">
            """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                cls = "chat-msg-bot" if msg['role'] == 'assistant' else "chat-msg-user"
                st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Quick questions
            st.markdown('<p class="pg-section-label" style="margin-top:10px;">Quick questions</p>',
                        unsafe_allow_html=True)
            q1, q2 = st.columns(2)
            quick_qs = [
                "Why is this URL flagged?",
                "Is it safe to visit?",
                "Could this be a false positive?",
                "What should I do next?",
            ]
            for i, q in enumerate(quick_qs):
                col = q1 if i % 2 == 0 else q2
                if col.button(q, key=f"quick_{q}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("PhishGuard AI thinking..."):
                        try:
                            resp = get_chatbot_response(
                                st.session_state.chat_url,
                                st.session_state.chat_result,
                                user_question=q,
                                chat_history=st.session_state.messages[:-1]
                            )
                            st.session_state.messages.append({"role": "assistant", "content": resp})
                        except Exception as e:
                            st.error(str(e))
                    st.rerun()

            user_q = st.chat_input("Type your question here...")
            if user_q:
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("PhishGuard AI thinking..."):
                    try:
                        resp = get_chatbot_response(
                            st.session_state.chat_url,
                            st.session_state.chat_result,
                            user_question=user_q,
                            chat_history=st.session_state.messages[:-1]
                        )
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        st.error(str(e))
                st.rerun()

        st.divider()

        # ── Features + Warnings ───────────────
        feat_col, warn_col = st.columns(2)

        with feat_col:
            st.markdown('<p class="pg-section-label">URL features</p>', unsafe_allow_html=True)
            display = [
                ("URL length",       f"{features['length_url']} chars",  "neutral"),
                ("Domain length",    f"{features['length_hostname']} chars", "neutral"),
                ("IP address",       "Yes" if features['ip'] else "No",
                 "danger" if features['ip'] else "safe"),
                ("@ symbol",         "Yes" if features['nb_at'] else "No",
                 "danger" if features['nb_at'] else "safe"),
                ("HTTPS",            "Yes" if features['https_token'] else "No",
                 "safe" if features['https_token'] else "danger"),
                ("Dot count",        str(features['nb_dots']),           "neutral"),
                ("Subdomain count",  str(features['nb_subdomains']),     "neutral"),
                ("Phish keywords",   str(features['phish_hints']),
                 "danger" if features['phish_hints'] > 0 else "safe"),
                ("Slash count",      str(features['nb_slash']),          "neutral"),
                ("Hyphen in domain", "Yes" if features['prefix_suffix'] else "No",
                 "danger" if features['prefix_suffix'] else "safe"),
            ]
            rows_html = ""
            for label, val, cls in display:
                rows_html += f"""
                <div class="pg-feature-row">
                    <span class="pg-feature-label">{label}</span>
                    <span class="pg-feature-val-{cls}">{val}</span>
                </div>"""
            st.markdown(
                f'<div class="pg-card" style="margin-bottom:0;">{rows_html}</div>',
                unsafe_allow_html=True
            )

        with warn_col:
            st.markdown('<p class="pg-section-label">Security warnings</p>', unsafe_allow_html=True)
            warnings_html = ""
            for w in result['warnings']:
                if "No suspicious" in w:
                    cls, ico = "pg-warning-safe",   "✓"
                elif any(x in w for x in ["@", "IP", "keyword", "Hyphen"]):
                    cls, ico = "pg-warning-danger",  "✕"
                else:
                    cls, ico = "pg-warning-medium",  "!"
                warnings_html += f"""
                <div class="pg-warning {cls}">
                    <span style='font-weight:600;'>{ico}</span>
                    <span>{w}</span>
                </div>"""
            st.markdown(
                f'<div class="pg-card" style="margin-bottom:0;">{warnings_html}</div>',
                unsafe_allow_html=True
            )
            if is_phish:
                st.error("Do not visit this website.")
            else:
                st.success("This URL appears safe to visit.")


# ════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ════════════════════════════════════════════
elif page == "📊  Model Dashboard":

    try:
        with open("models/evaluation_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("Run `python train.py` first to generate results.")
        st.stop()

    best_name = max(results, key=lambda x: results[x]['f1_score'])
    best      = results[best_name]

    st.markdown(f"""
    <div class="pg-metric-grid">
        <div class="pg-metric"><p class="pg-metric-value">{best['accuracy']}%</p><p class="pg-metric-label">Accuracy</p></div>
        <div class="pg-metric"><p class="pg-metric-value">{best['precision']}%</p><p class="pg-metric-label">Precision</p></div>
        <div class="pg-metric"><p class="pg-metric-value">{best['recall']}%</p><p class="pg-metric-label">Recall</p></div>
        <div class="pg-metric"><p class="pg-metric-value">{best['f1_score']}%</p><p class="pg-metric-label">F1-Score</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pg-card">
        <p class="pg-card-title">Best model</p>
        <p style='font-size:0.9rem; font-weight:600; color:#1a1a2e; margin:0;'>{best_name}</p>
        <p style='font-size:0.78rem; color:#6c757d; margin:4px 0 0;'>
            Trained on 8,196 samples · Tested on 2,049 samples
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pg-card">', unsafe_allow_html=True)
    st.markdown('<p class="pg-card-title">All models comparison</p>', unsafe_allow_html=True)
    rows = []
    for name, m in results.items():
        rows.append({
            'Model':     name + ("  ★ Best" if name == best_name else ""),
            'Accuracy':  f"{m['accuracy']}%",
            'Precision': f"{m['precision']}%",
            'Recall':    f"{m['recall']}%",
            'F1-Score':  f"{m['f1_score']}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    viz_dir   = "visualizations"
    chart_map = {
        "Model comparison":   "model_comparison.png",
        "Feature importance": "feature_importance.png",
        "Class distribution": "class_distribution.png",
    }
    for title_c, fname in chart_map.items():
        path = os.path.join(viz_dir, fname)
        if os.path.exists(path):
            st.markdown(f'<div class="pg-card"><p class="pg-card-title">{title_c}</p></div>', unsafe_allow_html=True)
            st.image(path, use_column_width=True)

    st.markdown('<p class="pg-section-label" style="margin-top:1rem;">Confusion matrices</p>', unsafe_allow_html=True)
    cm_cols = st.columns(2)
    for i, name in enumerate(results.keys()):
        safe = name.replace(' ', '_').lower()
        path = os.path.join(viz_dir, f"cm_{safe}.png")
        if os.path.exists(path):
            cm_cols[i % 2].image(path, caption=name, use_column_width=True)


# ════════════════════════════════════════════
# PAGE 3 — HOW IT WORKS
# ════════════════════════════════════════════
elif page == "📚  How It Works":

    c1, c2 = st.columns(2)
    cards = [
        ("🗂️", "Dataset",
         "11,430 real-world URLs from the GregaVrbancic Phishing Dataset. "
         "10,245 after removing 1,185 duplicates. Perfectly balanced — 50% phishing, 50% legitimate."),
        ("⚙️", "Feature engineering",
         "15 URL features selected from 88 available. Based on real-time extractability "
         "and discriminative power — without accessing the website."),
        ("🧠", "Random Forest model",
         "100 decision trees trained on 8,196 samples. Each tree votes independently. "
         "Majority vote gives the final prediction — 90.09% accuracy on 2,049 test samples."),
        ("🤖", "PhishGuard AI chatbot",
         "Groq API with Llama 3.3 explains why a URL is flagged. Handles false positives "
         "intelligently and answers follow-up questions in plain English."),
    ]
    for i, (icon, title_c, body) in enumerate(cards):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div class="pg-info-card" style='margin-bottom:1rem;'>
            <span style='font-size:1.3rem;'>{icon}</span>
            <p class="pg-info-title">{title_c}</p>
            <p class="pg-info-body">{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="pg-card" style="margin-top:0.5rem;">', unsafe_allow_html=True)
    st.markdown('<p class="pg-card-title">Detection pipeline</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pg-pipeline">
        <div class="pg-step">URL input</div>
        <span class="pg-arrow">→</span>
        <div class="pg-step">Extract 15 features</div>
        <span class="pg-arrow">→</span>
        <div class="pg-step">StandardScaler</div>
        <span class="pg-arrow">→</span>
        <div class="pg-step pg-step-active">Random Forest</div>
        <span class="pg-arrow">→</span>
        <div class="pg-step">Verdict + confidence</div>
        <span class="pg-arrow">→</span>
        <div class="pg-step">AI explanation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="pg-card">', unsafe_allow_html=True)
    st.markdown('<p class="pg-card-title">Top 3 features</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; flex-direction:column; gap:10px;'>
        <div style='display:flex; gap:12px; align-items:flex-start;'>
            <span style='font-size:0.8rem; font-weight:600; color:#1a73e8; min-width:90px;'>page_rank</span>
            <span style='font-size:0.82rem; color:#6c757d; line-height:1.5;'>
                Google trust score. Phishing sites average 1.89 vs 4.48 for legitimate sites.
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:flex-start;'>
            <span style='font-size:0.8rem; font-weight:600; color:#1a73e8; min-width:90px;'>phish_hints</span>
            <span style='font-size:0.82rem; color:#6c757d; line-height:1.5;'>
                Suspicious keyword count. Phishing URLs contain 12x more keywords like login, verify, secure.
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:flex-start;'>
            <span style='font-size:0.8rem; font-weight:600; color:#1a73e8; min-width:90px;'>domain_age</span>
            <span style='font-size:0.82rem; color:#6c757d; line-height:1.5;'>
                Domain registration age. Legitimate sites average 5,093 days vs phishing 3,031 days.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)