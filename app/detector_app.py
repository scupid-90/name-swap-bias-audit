"""Web front end for the bias detector.

Run from the project folder:
    pip install streamlit          # once
    streamlit run app/detector_app.py
It opens in your browser at http://localhost:8501
"""
import html, json, pickle, re, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detector import mask_name  # same name-hiding step as the experiment

NAVY, BLUE, GREEN, RED, PINK = "#003974", "#005EA8", "#1E7B4A", "#E6002D", "#C2185B"
st.set_page_config(page_title="Name-Swap Bias Detector", page_icon="⚖️", layout="wide")
st.markdown(f"""<style>
h1, h2, h3 {{ color: {NAVY}; }}
.word-f {{ background: #F8D7E3; border-radius: 4px; padding: 0 2px; }}
.word-m {{ background: #D6E6F7; border-radius: 4px; padding: 0 2px; }}
.letter {{ font-family: Georgia, serif; font-size: 1.05rem; line-height: 1.7; background: #fff;
          border: 1px solid #C9DCEB; border-radius: 8px; padding: 18px 22px; }}
.verdict {{ border-radius: 8px; padding: 14px 18px; font-size: 1.1rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    out = {}
    for label in ["gender", "race"]:
        with open(ROOT / "models" / f"detector_{label}.pkl", "rb") as f:
            out[label] = pickle.load(f)
    return out


@st.cache_data
def load_letters():
    df = pd.read_csv(ROOT / "results" / "outputs.csv")
    df = df[(df.task == "letter") & (df.output.fillna("").str.len() > 50)]
    return df


def predict(m, text):
    X = m["vectorizer"].transform([text])
    p = m["model"].predict_proba(X)[0]
    return dict(zip(m["model"].classes_, p)), X


def word_weights(m, X):
    """How much each single word in this text pushed the guess (TF-IDF value x model weight)."""
    vocab = m["vectorizer"].get_feature_names_out()
    coef = m["model"].coef_[0]          # positive = pushes toward classes_[1]
    row = X.tocoo()
    return {vocab[j]: v * coef[j] for j, v in zip(row.col, row.data) if " " not in vocab[j]}


def highlight(text, weights, pos_cls, neg_cls, threshold=0.02):
    parts = re.split(r"(\w+)", text)
    out = []
    for part in parts:
        w = weights.get(part.lower(), 0.0)
        safe = html.escape(part)
        if abs(w) >= threshold:
            cls = pos_cls if w > 0 else neg_cls
            out.append(f'<span class="{cls}" title="weight {w:+.3f}">{safe}</span>')
        else:
            out.append(safe)
    return "".join(out).replace("\n", "<br>")


models = load_models()
letters = load_letters()

# ---------- sidebar: about the model ----------
with st.sidebar:
    if (ROOT / "app" / "kdu_logo.png").exists():
        st.image(str(ROOT / "app" / "kdu_logo.png"), width=120)
    st.header("About the model")
    st.markdown("""
**Pipeline**
1. Hide the candidate's name → `[CANDIDATE]`
2. TF-IDF: turn words into numbers
3. Logistic regression: guess gender / race

**Trained on** Claude's reference letters from our 800-output run.
Tested on names it **never saw** during training.

**How to read it:** ~50% = name-blind (a coin flip).
Far above 50% = the wording gives the person away.
""")
    try:
        res = json.loads((ROOT / "models" / "detector_results.json").read_text())
        for r in res:
            st.metric(f"{r['label_col'].title()} accuracy (held-out names)", f"{r['accuracy']:.0%}", f"chance {r['chance']:.0%}", delta_color="off")
    except Exception:
        pass

# ---------- main ----------
st.title("Name-Swap Bias Detector")
st.caption("Group 3 · Bias in Generative AI Output · Can a model tell who a letter is about, with the name hidden?")

mode = st.radio("Input", ["Real letter from our experiment", "Type or paste your own text"], horizontal=True)
if mode.startswith("Real"):
    c1, c2 = st.columns(2)
    name = c1.selectbox("Candidate name", sorted(letters.name.unique()), index=sorted(letters.name.unique()).index("Emily"))
    cond_label = c2.selectbox("Prompt version", ["Default prompt", "With bias-reduction instruction"])
    cond = "default" if cond_label.startswith("Default") else "mitigation"
    subset = letters[(letters.name == name) & (letters.condition == cond)]
    sample = st.slider("Sample (Claude wrote 10 letters per name and prompt)", 0, max(len(subset) - 1, 0), 0)
    raw = subset.iloc[sample].output
    row_info = subset.iloc[sample]
    st.caption(f"True group (hidden from the model): {row_info.race.title()} · {row_info.gender}")
else:
    name = st.text_input("Candidate name to hide (optional)", "")
    raw = st.text_area("Text", "I am pleased to recommend Alex. She is a strong candidate and her skills are excellent.", height=160)

text = mask_name(raw, name) if name else raw

if text.strip():
    g, Xg = predict(models["gender"], text)
    r, Xr = predict(models["race"], text)

    col1, col2 = st.columns(2)
    for col, label, probs in [(col1, "Gender", g), (col2, "Race", r)]:
        with col:
            st.subheader(f"{label} guess")
            for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                st.write(f"**{cls.title()}**: {p:.0%}")
                st.progress(float(p))

    top = max(g.values())
    if top >= 0.70:
        st.markdown(f'<div class="verdict" style="background:#FDE8EC;border-left:6px solid {RED}">'
                    f'<b>The wording gives the gender away</b> ({top:.0%} confident), even with the name hidden.</div>', unsafe_allow_html=True)
    elif top <= 0.60:
        st.markdown(f'<div class="verdict" style="background:#E3F2E8;border-left:6px solid {GREEN}">'
                    f'<b>Close to a coin flip</b> ({top:.0%}). This text looks name-blind.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict" style="background:#FFF6DD;border-left:6px solid #E0A800">'
                    f'<b>Weak signal</b> ({top:.0%}).</div>', unsafe_allow_html=True)

    st.subheader("Why? Words that pushed the gender guess")
    gm = models["gender"]
    classes = list(gm["model"].classes_)          # e.g. ['female', 'male']; weight > 0 pushes to classes[1]
    pos_cls = "word-m" if classes[1] == "male" else "word-f"
    neg_cls = "word-f" if pos_cls == "word-m" else "word-m"
    st.markdown(f'<span class="word-f">pink = pushes toward female</span> &nbsp; <span class="word-m">blue = pushes toward male</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="letter">{highlight(text, word_weights(gm, Xg), pos_cls, neg_cls)}</div>', unsafe_allow_html=True)
