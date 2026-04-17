# Email Spam Classification using Deep Learning

> A comparative study of classical ML and modern deep learning approaches for email spam detection, built on the SpamAssassin corpus.

**Authors:** Muhammad Hamza Nawaz · Muhammad Ibrahim · Rameela Hassan  
**Institution:** Department of Artificial Intelligence, FAST-NUCES Karachi  
**Course:** Natural Language Processing

---

## Overview

This project presents a deep learning–centric framework for email spam classification that advances beyond classical ML baselines. We evaluate a full model progression — from Logistic Regression and SVM through ANN, BiLSTM, BERT, and a weighted ensemble — on the **SpamAssassin Public Corpus**, a real-world dataset containing raw email bodies, headers, HTML content, and MIME structures.

The work critically revisits Iqbal & Khan (2025), who benchmarked classical ML on the outdated UCI Spambase (1998) dataset, and demonstrates that raw-text–based deep learning models generalize substantially better to modern spam and phishing threats.

---

## Key Results

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 0.978    | 1.000     | 0.939  | 0.968 |
| Naïve Bayes         | 0.966    | 1.000     | 0.903  | 0.949 |
| SVM (RBF)           | 0.985    | 0.996     | 0.960  | 0.978 |
| Random Forest       | 0.985    | 0.993     | 0.964  | 0.978 |
| ANN                 | 0.992    | 0.996     | 0.982  | 0.989 |
| BiLSTM              | 0.964    | 0.970     | 0.928  | 0.949 |
| **BERT**            | **0.992**| **1.000** | **0.979** | **0.990** |
| Ensemble            | 0.982    | 1.000     | 0.951  | 0.975 |

All metrics are computed on a held-out 20% stratified test split (900 emails).

---

## Dataset

**SpamAssassin Public Corpus**

- Spam emails: 1,897
- Ham emails: 2,600
- Total: 4,497 emails

Unlike the numeric-only UCI Spambase corpus, SpamAssassin includes raw email text, headers, HTML/plain-text formats, embedded URLs, forged sender identities, and MIME multipart structures — making it substantially more representative of real-world threat patterns.

---

## Methodology

### Preprocessing

- HTML tag stripping (retaining text content)
- URL and domain preservation (key phishing indicators)
- Unicode and whitespace normalization
- Lowercasing applied for TF-IDF models; disabled for BERT tokenization
- Stopwords intentionally retained (contextually informative phrases like "click here" and "verify your account")

### Feature Representations

| Model Type     | Representation                        |
|----------------|---------------------------------------|
| Classical ML / ANN | TF-IDF (20k features, 1–3 n-grams) |
| BiLSTM         | FastText 300-D embeddings, max 512 tokens |
| BERT           | WordPiece tokenization, max 256 tokens |

### Models

**Classical Baselines**
- Logistic Regression (L2 regularization)
- SVM with RBF kernel
- Random Forest

**Deep Learning**
- ANN: Dense(512, ReLU) → Dense(128, ReLU) → Dropout → Softmax
- BiLSTM: 300-D FastText embeddings, 64 units per direction, 0.4 dropout
- BERT: `bert-base-uncased`, fine-tuned for 3 epochs at lr=2e-5, batch size=16

**Ensemble**

Weighted fusion of the three deep models:

```
ŷ = 0.60 · ŷ_BERT + 0.25 · ŷ_BiLSTM + 0.15 · ŷ_ANN
```

Weights were set heuristically based on individual model validation performance, with BERT receiving the highest weight given its superior standalone results.

---

## Why Deep Learning Outperforms Classical ML

Classical models (SVM, LR, Naïve Bayes) represent text as sparse TF-IDF vectors, ignoring word order and semantic context. Deep architectures address this:

- **BERT** captures contextual phrase semantics (e.g., "verify your account," "security update"), sender-body relationships, and nuanced linguistic patterns
- **BiLSTM** models sequential flow and persuasive narrative patterns in phishing emails
- **ANN** detects high-dimensional TF-IDF lexical signals efficiently

The ensemble leverages complementary strengths of all three, reducing both false positives and false negatives.

---

## Limitations

- **Dataset size**: 4,497 emails is small relative to production-scale corpora; results may not fully generalize
- **English only**: No multilingual or code-switched email support
- **Adversarial robustness**: Models were not tested against adversarial perturbations or obfuscation attacks
- **Static ensemble weights**: Weights were manually set; learned stacking may yield further gains
- **Temporal drift**: Spam patterns evolve; periodic retraining would be required in deployment

---

## Setup and Reproducibility

### Requirements

```
Python 3.10
PyTorch 2.2
transformers
scikit-learn
nltk
beautifulsoup4
fasttext
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Hardware Used

- NVIDIA RTX 4060 (8GB VRAM)
- Intel Core i5
- 32GB RAM
- CUDA 12

### Project Structure

```
├── data/
│   └── spamassassin/          # Raw email files
├── preprocessing/
│   └── parse_emails.py        # Email parsing and cleaning
├── models/
│   ├── classical_ml.py        # LR, SVM, RF with TF-IDF
│   ├── ann.py                 # ANN model
│   ├── bilstm.py              # BiLSTM model
│   └── bert_finetune.py       # BERT fine-tuning pipeline
├── ensemble/
│   └── fusion.py              # Weighted ensemble inference
├── evaluation/
│   └── metrics.py             # Accuracy, F1, ROC-AUC, confusion matrix
└── README.md
```

---

## Future Directions

- Multilingual transformer models (XLM-R, mBERT) for non-English spam
- Adversarial training against lexical obfuscation and template mutations
- URL reputation scoring and domain similarity features
- Lightweight deployment via DistilBERT
- Learned ensemble stacking (meta-learner) instead of fixed weights
- Graph neural networks for sender–receiver relationship modeling

---

## References

1. Iqbal, M. & Khan, S. — *Email Classification Analysis Using Machine Learning Techniques*, Applied Computing and Informatics, 2025
2. Devlin, J. et al. — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL-HLT, 2019
3. Hochreiter, S. & Schmidhuber, J. — *Long Short-Term Memory*, Neural Computation, 1997
4. Apache Software Foundation — *SpamAssassin Public Corpus*, 2023
5. Bojanowski, P. et al. — *Enriching Word Vectors with Subword Information*, TACL, 2017
6. Breiman, L. — *Random Forests*, Machine Learning, 2001
7. Cortes, C. & Vapnik, V. — *Support-Vector Networks*, Machine Learning, 1995
