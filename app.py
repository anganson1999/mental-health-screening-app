import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "mental_health_bert_model"
MAX_LENGTH = 256

st.set_page_config(page_title="Mental Health Text Screening", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    """Return a list of (label, probability) sorted by probability, descending."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    id2label = model.config.id2label
    ranked = sorted(
        ((id2label[i], float(probs[i])) for i in range(len(probs))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked


def main():
    st.title("🧠 Text-Based Mental Health Screening")
    st.caption("NLP Group Project — BERT fine-tuned text classifier (transformer-based)")

    st.warning(
        "This tool is an **academic prototype** for research/demo purposes only. "
        "It is not a medical diagnosis and should never replace professional mental "
        "health care. If you or someone you know is in crisis, please contact a "
        "local emergency service or a crisis helpline immediately."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    tokenizer, model = load_model()

    tab_screen, tab_history, tab_about = st.tabs(["Screening", "History", "About"])

    with tab_screen:
        text = st.text_area(
            "Enter text to screen",
            height=150,
            placeholder="Describe how you've been feeling lately...",
        )
        if st.button("Analyze", type="primary"):
            if not text.strip():
                st.error("Please enter some text first.")
            else:
                ranked = predict(text, tokenizer, model)
                top_label, top_conf = ranked[0]
                st.subheader(f"Predicted class: **{top_label}**  ({top_conf:.1%} confidence)")

                for label, prob in ranked: st.write(f"{label} - {prob:.1%}"); st.progress(min(max(prob, 0.0), 1.0))
                st.session_state.history.append(
                    {"text": text, "prediction": top_label, "confidence": f"{top_conf:.1%}"}
                )

    with tab_history:
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No screenings yet this session.")

    with tab_about:
        st.markdown(
            """
            **Model**: BERT (`bert-base-uncased`) fine-tuned for single-label text
            classification across 7 mental-health-related classes: anxiety, bipolar,
            depression, normal, personality disorder, stress, suicidal.

            **Project**: NLP group project on transformer-based text classification.

            **Responsible use**: predictions are statistical estimates from a
            classroom project model, not a clinical assessment. Please treat results
            as informational only.
            """
        )


if __name__ == "__main__":
    main()
