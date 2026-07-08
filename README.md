# Text-Based Mental Health Disorder Classification

Streamlit app for an NLP group project: a BERT (transformer) model fine-tuned to
classify free-text into mental-health-related categories.

## Target classes

- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

## Project structure

```
app.py                      Streamlit app (loads the model + serves the UI)
requirements.txt            Python dependencies
mental_health_bert_model/   Fine-tuned BERT weights, tokenizer, and config (tracked with Git LFS)
notebooks/                  Training / model-comparison notebooks

```

## Run locally

```bash
git lfs install
git clone <this-repo-url>
cd mental-health-screening-app
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected
directly to this GitHub repo (`main` branch, `app.py` entry point).

## Important note

This app is an academic prototype / screening-support demo only. It is **not**
a medical diagnosis tool and should not replace professional mental health care.


## Data source

The fine-tuning dataset used for this project's BERT model is the [Mental_Health_Condition_Classification](https://huggingface.co/datasets/sai1908/Mental_Health_Condition_Classification) dataset, published on Hugging Face by sai1908. Credit to the original author for compiling and sharing this dataset.
