"""
wordCom Ai — LSTM Sentence Completion
A Streamlit app that loads a trained LSTM model + tokenizer and completes
whatever sentence the user starts typing. Brand name and developer credit
are set via APP_NAME / DEVELOPER_NAME below — change them anytime.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Place these three files in the same folder as this script:
    lstm_model(1).h5
    tokenizer.pkl
    max_len.pkl
"""

import os
import pickle

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Brand / developer config — edit this whenever you land on a product name
# ----------------------------------------------------------------------------
APP_NAME = "wordCom Ai"  # <- swap this out anytime, everything below follows it
APP_ICON = "💬"  # word/communication themed icon — used in tab icon + header
DEVELOPER_NAME = "Suraj Jadhav"  # shown only as a small footer credit

# ----------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Sentence Completion",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# File paths — edit these if your filenames differ
# ----------------------------------------------------------------------------
MODEL_PATH = "lstm_model(1).h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN_PATH = "max_len.pkl"

# ----------------------------------------------------------------------------
# Styling — dark, glassy, glowing "AI lab" theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    :root{
        --bg:#0B1120;
        --card:#111827;
        --primary:#00E5FF;
        --secondary:#6C63FF;
        --accent:#00FFB3;
        --text:#F9FAFB;
    }

    html, body, [class*="css"]  { font-family:'Inter', sans-serif; }

    /* Hide scrollbars everywhere, on every element — scrolling still works
       via mouse wheel / trackpad / keyboard, it's just not drawn. */
    *{
        scrollbar-width: none;        /* Firefox */
        -ms-overflow-style: none;     /* old Edge / IE */
    }
    *::-webkit-scrollbar{
        width: 0px;
        height: 0px;
        background: transparent;      /* Chrome / Safari / new Edge */
    }

    /* Streamlit reserves a lot of empty space above the content by default —
       trim it to a small, deliberate gap so the header sits just below the
       top edge instead of being glued to it or floating in a huge void. */
    .block-container{
        padding-top: 2.25rem !important;
        padding-bottom: 1rem !important;
        max-width: 1100px;
    }
    hr{ margin: 0.5rem 0 !important; }

    .stApp{
        background:
            radial-gradient(circle at 15% 10%, rgba(108,99,255,0.20), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(0,229,255,0.16), transparent 45%),
            radial-gradient(circle at 50% 100%, rgba(0,255,179,0.10), transparent 40%),
            var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"]{
        background: rgba(17,24,39,0.85);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero{
        text-align:center;
        padding: 0.75rem 1rem 0.25rem 1rem;
    }
    .glow-title{
        font-family:'Poppins', sans-serif;
        font-weight:800;
        font-size: 2.3rem;
        margin-bottom:0.15rem;
        background: linear-gradient(90deg, var(--primary), var(--secondary) 55%, var(--accent));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 0 40px rgba(0,229,255,0.25);
        letter-spacing: -0.5px;
    }
    .subtitle{
        color: rgba(249,250,251,0.65);
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* Custom-styled containers — st.container(key=...) — become glass cards
       purely via this CSS (we don't use Streamlit's own border=True, which
       draws a second competing box on top of ours). One box, not two. */
    .st-key-input_card,
    .st-key-input_card > div{
        background: rgba(17,24,39,0.55) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .st-key-input_card{ padding: 0.5rem 0.6rem 0.7rem 0.6rem; }

    .st-key-history_panel,
    .st-key-history_panel > div{
        background: rgba(17,24,39,0.5) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
    }
    .st-key-history_panel{ padding: 0.6rem 1rem; }

    /* About card — gradient-bordered glass panel with a glowing icon badge */
    .about-card{
        position: relative;
        background: rgba(17,24,39,0.6);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.30);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .about-card::before{
        content: "";
        position: absolute;
        inset: 0;
        padding: 1px;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.55;
        pointer-events: none;
    }
    .about-card:hover{
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0,229,255,0.18);
    }
    .about-card:hover::before{
        opacity: 0.9;
    }
    .about-icon{
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        font-size: 1.2rem;
        margin-bottom: 0.55rem;
        box-shadow: 0 0 18px rgba(0,229,255,0.35);
    }
    .about-title{
        font-family:'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text);
        margin-bottom: 0.35rem;
        letter-spacing: -0.2px;
    }
    .about-text{
        font-size: 0.76rem;
        line-height: 1.5;
        color: rgba(249,250,251,0.62);
        margin: 0;
    }
    /* The text area blends into its card instead of drawing its own separate
       box — no border of its own, just a soft inset glow when focused. */
    div[data-testid="stTextArea"] textarea{
        background: transparent;
        border: none;
        border-radius: 12px;
        color: var(--text);
        font-size: 1.05rem;
        padding: 0.6rem 0.7rem;
        transition: box-shadow 0.15s ease, background 0.15s ease;
    }
    div[data-testid="stTextArea"] textarea:focus{
        box-shadow: 0 0 0 2px rgba(0,229,255,0.35) inset;
        background: rgba(255,255,255,0.02);
        outline: none;
    }
    div[data-testid="stTextArea"] textarea::placeholder{
        color: rgba(249,250,251,0.35);
    }

    .result-card{
        margin-top: 0.6rem;
        background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(108,99,255,0.10));
        border: 1px solid rgba(0,229,255,0.35);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        box-shadow: 0 0 35px rgba(0,229,255,0.12);
    }
    .result-label{
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.72rem;
        color: var(--accent);
        font-family:'JetBrains Mono', monospace;
        margin-bottom: 0.4rem;
    }
    .result-text{
        font-size: 1.35rem;
        font-weight: 500;
        line-height: 1.6;
        color: var(--text);
    }

    div.stButton > button{
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        color: var(--text);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover{
        border-color: var(--primary);
        color: var(--primary);
        box-shadow: 0 0 18px rgba(0,229,255,0.25);
    }
    div.stButton > button[kind="primary"]{
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border: none;
        color: #061018;
        font-weight: 700;
    }
    div.stButton > button[kind="primary"]:hover{
        box-shadow: 0 0 25px rgba(0,229,255,0.45);
        transform: translateY(-1px);
    }

    /* Floating action buttons — Download + History. Rounded pills that sit
       up off the page with real elevation, and lift further on hover. */
    div[data-testid="stDownloadButton"] button,
    .st-key-history_toggle button{
        border-radius: 999px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.40);
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    div[data-testid="stDownloadButton"] button:hover,
    .st-key-history_toggle button:hover{
        transform: translateY(-3px);
    }

    /* Download — cyan/purple gradient, the "positive" primary-feeling action */
    div[data-testid="stDownloadButton"] button{
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border: none;
        color: #061018;
    }
    div[data-testid="stDownloadButton"] button:hover{
        box-shadow: 0 14px 30px rgba(0,229,255,0.45);
    }

    /* History — solid mint fill, reads as the secondary action */
    .st-key-history_toggle button{
        background: rgba(0,255,179,0.16) !important;
        border: 1px solid rgba(0,255,179,0.55) !important;
        color: var(--accent) !important;
    }
    .st-key-history_toggle button:hover{
        background: rgba(0,255,179,0.26) !important;
        box-shadow: 0 14px 30px rgba(0,255,179,0.30);
    }

    [data-testid="stMetricValue"]{
        color: var(--primary);
        font-family:'JetBrains Mono', monospace;
    }

    .footer{
        text-align:center;
        margin-top: 2rem;
        padding: 1rem 0.5rem 0.25rem 0.5rem;
        color: rgba(249,250,251,0.4);
        font-size: 0.82rem;
        letter-spacing: 0.3px;
        font-family:'JetBrains Mono', monospace;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Cached loading of model + tokenizer + max_len
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    from tensorflow.keras.models import load_model  # local import: heavy, only load when needed

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAXLEN_PATH, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, int(max_len)


# ----------------------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------------------
def predict_next_words(model, tokenizer, max_len, seed_text, n_words):
    """Iteratively predicts n_words, appending each to the seed text.

    Always takes the single most-likely next word (greedy decoding) — no
    randomness, so the same input always produces the same completion.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    index_word = tokenizer.index_word
    generated = seed_text

    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
        preds = model.predict(token_list, verbose=0)[0]

        next_index = int(np.argmax(preds))
        next_word = index_word.get(next_index, "")
        if not next_word:
            break
        generated += " " + next_word

    return generated


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="glow-title">{APP_ICON} {APP_NAME}</div>
        <p class="subtitle">LSTM-powered sentence completion — start a thought, watch the model finish it</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load artifacts (with friendly errors instead of a crash)
# ----------------------------------------------------------------------------
missing = [p for p in [MODEL_PATH, TOKENIZER_PATH, MAXLEN_PATH] if not os.path.exists(p)]
if missing:
    st.error(
        "Missing file(s): **" + ", ".join(missing) + "**\n\n"
        "Place `lstm_model(1).h5`, `tokenizer.pkl`, and `max_len.pkl` in the same "
        "folder as `app.py`, then rerun the app."
    )
    st.stop()

try:
    with st.spinner("Loading model..."):
        model, tokenizer, max_len = load_artifacts()
except Exception as e:
    st.error(f"Couldn't load the model artifacts: {e}")
    st.stop()

vocab_size = len(tokenizer.word_index) + 1

# ----------------------------------------------------------------------------
# Sidebar — settings + model info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Generation settings")
    n_words = st.slider("Words to generate", 1, 30, 8)

    st.markdown("---")
    st.markdown("### 📊 Model info")
    c1, c2 = st.columns(2)
    c1.metric("Vocabulary", f"{vocab_size:,}")
    c2.metric("Max seq. len", max_len)
    st.markdown("---")

    st.markdown(
        """
        <div class="about-card">
            <div class="about-icon">💡</div>
            <div class="about-title">About wordCom Ai</div>
            <p class="about-text">
                A lightweight sentence-completion tool built on a custom-trained
                LSTM (Long Short-Term Memory) neural network. Start typing a
                thought, and it predicts what comes next — word by word — right
                in your browser, no sign-up or setup needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

# ----------------------------------------------------------------------------
# Main input area
# ----------------------------------------------------------------------------
st.markdown("#### Start typing — or try an example")

examples = [
    "Once upon a time",
    "The future of artificial intelligence",
    "In the middle of the night",
    "She opened the door and",
]

if "seed_text" not in st.session_state:
    st.session_state.seed_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_completion" not in st.session_state:
    st.session_state.last_completion = ""
if "show_history" not in st.session_state:
    st.session_state.show_history = False

ex_cols = st.columns(len(examples))
for col, ex in zip(ex_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.seed_text = ex

with st.container(key="input_card"):
    seed_text = st.text_area(
        "Your sentence",
        value=st.session_state.seed_text,
        height=80,
        placeholder="Type the beginning of a sentence...",
        label_visibility="collapsed",
    )
    generate = st.button("✨ Generate completion", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Run prediction
# ----------------------------------------------------------------------------
if generate:
    if not seed_text.strip():
        st.warning("Type something first!")
    else:
        with st.spinner("Thinking..."):
            completion = predict_next_words(
                model, tokenizer, max_len, seed_text.strip(), n_words
            )
        st.session_state.history.insert(0, completion)
        st.session_state.last_completion = completion
        st.session_state.show_history = False
        st.balloons()

# ----------------------------------------------------------------------------
# Result + actions — stays visible across reruns (e.g. clicking History)
# since it reads from session_state rather than the one-shot `generate` flag.
# ----------------------------------------------------------------------------
if st.session_state.last_completion:
    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-label">Completed sentence</p>
            <p class="result-text">{st.session_state.last_completion}</p>
        </div>
        <br><br>
        """,
        unsafe_allow_html=True,
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.download_button(
            "⬇️ Download",
            st.session_state.last_completion,
            file_name="completion.txt",
            use_container_width=True,
        )
    with bcol2:
        if st.button(
            f"📜 History ({len(st.session_state.history)})",
            key="history_toggle",
            use_container_width=True,
        ):
            st.session_state.show_history = not st.session_state.show_history

    if st.session_state.show_history:
        with st.container(key="history_panel"):
            if st.session_state.history:
                for i, h in enumerate(st.session_state.history):
                    st.markdown(f"**{i + 1}.** {h}")
                if st.button("Clear history", use_container_width=True):
                    st.session_state.history = []
                    st.session_state.show_history = False
                    st.rerun()
            else:
                st.caption("No history yet.")

st.markdown(
    f'<div class="footer">{APP_ICON} {APP_NAME} · built by {DEVELOPER_NAME} · TensorFlow · Keras LSTM · Streamlit</div>',
    unsafe_allow_html=True,
)