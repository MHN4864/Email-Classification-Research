import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from utils import clean_text, encode_bilstm
from models import ANNClassifier, BiLSTMClassifier, BERTClassifier
from scipy.special import expit
import os
import gdown

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODELS =================
@st.cache_resource
def load_everything():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    classical = {}
    for name in ["logistic_regression", "naive_bayes", "svm", "random_forest"]:
        with open(f"models/best_{name}.pkl", "rb") as f:
            classical[name] = pickle.load(f)

    ann = ANNClassifier(tfidf.max_features).to(DEVICE)
    ann.load_state_dict(torch.load("models/best_ann.pth")["model_state_dict"])
    ann.eval()

    with open("models/bilstm_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    bilstm = BiLSTMClassifier(len(vocab)).to(DEVICE)
    bilstm.load_state_dict(torch.load("models/best_bilstm.pth")["model_state_dict"])
    bilstm.eval()

    BERT_MODEL_PATH = "models/best_bert.pth"
    FILE_ID = "10l6HOSAgpwfEsKHRHnzEVomjhob7nDR-"

    if not os.path.exists(BERT_MODEL_PATH) or os.path.getsize(BERT_MODEL_PATH) < 100_000_000:
        os.makedirs("models", exist_ok=True)
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            BERT_MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    bert = BERTClassifier().to(DEVICE)
    checkpoint = torch.load(BERT_MODEL_PATH, map_location=DEVICE)
    bert.load_state_dict(checkpoint["model_state_dict"])
    bert.eval()

    return tfidf, classical, ann, bilstm, bert, tokenizer, vocab

tfidf, classical_models, ann, bilstm, bert, tokenizer, vocab = load_everything()

# ================= UI =================
st.set_page_config(page_title="Email Spam Classifier", layout="wide")
st.title("📧 Multi-Model Email Spam Classifier")

col1, col2 = st.columns(2)

with col1:
    sender = st.text_input("Sender (From)")
    subject = st.text_input("Subject")

with col2:
    body = st.text_area("Email Body", height=200)

# ================= PREDICT =================
if st.button("Classify Email"):
    combined = f"{sender} {subject} {body}"
    cleaned = clean_text(combined)

    results = []

    with torch.no_grad():
        # Classical ML
        X_tfidf = tfidf.transform([cleaned])

        for name, model in classical_models.items():
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_tfidf)[0][1]
            else:
                score = model.decision_function(X_tfidf)[0]
                prob = expit(score)

            pred = int(prob > 0.5)
            results.append([name.replace("_", " ").title(), pred, prob])

        # ANN
        X_ann = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(DEVICE)
        prob = torch.softmax(ann(X_ann), dim=1)[0][1].item()
        results.append(["ANN", int(prob > 0.5), prob])

        # BiLSTM
        X_lstm = encode_bilstm(cleaned, vocab).to(DEVICE)
        prob = torch.softmax(bilstm(X_lstm), dim=1)[0][1].item()
        results.append(["Bi-LSTM", int(prob > 0.5), prob])

        # BERT
        enc = tokenizer(
            cleaned,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        prob = torch.softmax(
            bert(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)),
            dim=1
        )[0][1].item()
        results.append(["BERT", int(prob > 0.5), prob])

    # ================= DISPLAY =================
    df = pd.DataFrame(results, columns=["Model", "Prediction", "Spam Probability"])
    df["Prediction"] = df["Prediction"].map({0: "Ham", 1: "Spam"})

    st.subheader("📊 Model Predictions")
    st.dataframe(df, use_container_width=True)

    # ================= ENSEMBLE =================
    ensemble_prob = df["Spam Probability"].mean()
    ensemble_pred = "Spam 🚨" if ensemble_prob > 0.5 else "Ham ✅"

    st.subheader("🧠 Ensemble Result")
    st.metric(
        label="Final Prediction",
        value=ensemble_pred,
        delta=f"{ensemble_prob:.2f} Spam Probability"
    )
