# Sentiment Analysis & Sector-Based Auto Reply System using NLP

An NLP pipeline that classifies customer feedback by **sentiment** (Positive / Negative / Neutral) and **industry sector** (Banking / Telecom / E-commerce), then generates context-aware automated replies.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [NLP Pipeline](#nlp-pipeline)
4. [Sector Detection](#sector-detection)
5. [Auto Reply System](#auto-reply-system)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Output & Visualisations](#output--visualisations)
9. [Project Structure](#project-structure)
10. [Dependencies](#dependencies)

---

## Project Overview

This system processes free-text customer reviews and produces:

- **Sentiment classification** — Positive, Negative, or Neutral
- **Sector identification** — Banking, Telecom, or E-commerce
- **Automated reply** — a canned response tailored to the predicted sentiment
- **Session visualisations** — sentiment pie chart and per-sector word clouds

| Metric | Value |
|---|---|
| Training samples | 999 |
| Sectors | 3 (Banking, Telecom, E-commerce) |
| Sentiment classes | 3 (Positive, Negative, Neutral) |
| Sector keywords | 182 total |
| Vectorizer features | 6 000 (TF-IDF, 1–2 grams) |
| Model | Logistic Regression (`max_iter=400`) |

---

## Dataset

### `advanced_nlp_dataset.csv`

| Column | Type | Description |
|---|---|---|
| `Sector` | `str` | E-commerce, Banking, or Telecom |
| `Text` | `str` | Raw customer review (1–3 sentences) |
| `Sentiment` | `str` | Positive, Negative, or Neutral — the target label |

**Label distribution (balanced):**

```
Positive  ≈ 333 rows  ████████████████████
Negative  ≈ 333 rows  ████████████████████
Neutral   ≈ 333 rows  ████████████████████
```

**Sample record:**

```csv
Sector,Text,Sentiment
E-commerce,"I had a genuinely excellent experience with the E-commerce service
where the overall process was smooth, the staff or system was responsive,
and everything worked exactly as expected without any inconvenience.",Positive
```

---

### `expanded_sector_keywords.csv`

| Column | Description |
|---|---|
| `Sector` | One of: Banking, Telecom, E-commerce |
| `Keyword` | Single token matched against preprocessed input |

**Keyword counts:**

| Sector | Keywords |
|---|---|
| Banking | 64 |
| Telecom | 60 |
| E-commerce | 58 |

Sample keywords per sector:

- **Banking** — `bank`, `account`, `savings`, `deposit`, `transaction`, `loan`, `credit`, `debit`, `balance`, `transfer`, `atm`, `mortgage`…
- **Telecom** — `network`, `signal`, `data`, `roaming`, `sim`, `internet`, `wifi`, `broadband`, `recharge`, `connection`, `speed`…
- **E-commerce** — `order`, `delivery`, `shipping`, `refund`, `return`, `cart`, `checkout`, `payment`, `product`, `tracking`, `seller`…

---

## NLP Pipeline

```
Raw text
  │
  ▼
Lowercase + strip punctuation (string.punctuation)
  │
  ▼
Remove stopwords (NLTK English stopword list)
  │
  ▼
Lemmatize (WordNetLemmatizer)
  │
  ▼
Synonym expansion (up to 2 WordNet synonyms per token)
  │
  ▼
TF-IDF vectorization (ngram_range=(1,2), max_features=6000)
  │
  ▼
Logistic Regression (max_iter=400, 80/20 stratified split)
  │
  ▼
Sentiment prediction + class probabilities
```

### Step-by-step

**1. Text cleaning**
Converts input to lowercase and strips all characters found in `string.punctuation`.

**2. Stopword removal**
NLTK's English stopword list removes uninformative tokens (`the`, `is`, `at`, etc.).

**3. Lemmatization**
`WordNetLemmatizer` reduces inflected words to their base form (`running → run`, `banks → bank`).

**4. Synonym expansion**
Up to 2 WordNet synonyms per token are appended to enrich sparse vocabulary and improve generalisation.

**5. TF-IDF vectorization**
Builds a unigram + bigram vocabulary capped at 6 000 features. The vectorizer is fitted on training data only to prevent data leakage.

**6. Logistic Regression**
Trained on 80% of data (stratified split), evaluated on the remaining 20%. `predict_proba` returns confidence scores for all three classes.

---

## Sector Detection

Sector is determined by counting keyword matches between the preprocessed input and each sector's keyword list from `expanded_sector_keywords.csv`.

```python
sector_scores = {}
for sector, keywords in sector_dict.items():
    count = sum(1 for keyword in keywords if keyword in clean_text)
    sector_scores[sector] = count

best_sector = max(sector_scores, key=sector_scores.get)
if sector_scores[best_sector] == 0:
    return "Unknown"
```

If no keywords match, the sector is labelled **Unknown**.

---

## Auto Reply System

Rule-based response generation based on predicted sentiment:

| Sentiment | Reply |
|---|---|
| Positive | `"Thank you for your positive feedback! 😊"` |
| Negative | `"We are sorry for your experience 😔"` |
| Neutral | `"Thank you for your feedback."` |

---

## Installation

**1. Clone or download the repository**

```bash
git clone <repo-url>
cd <repo-folder>
```

**2. Install dependencies**

```bash
pip install pandas nltk scikit-learn wordcloud matplotlib
```

**3. Download NLTK data** (handled automatically on first run)

```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Usage

Ensure the following three files are in the same directory:

```
nlp_project_org.py
advanced_nlp_dataset.csv
expanded_sector_keywords.csv
```

**Run the script:**

```bash
python nlp_project_org.py
```

**Interactive session:**

```
Enter review (type 'exit'): My internet keeps dropping every hour, very frustrating

 Input: My internet keeps dropping every hour, very frustrating
 Sector: Telecom
 Sentiment: Negative

 Probabilities:
 Negative: 0.761
 Neutral:  0.153
 Positive: 0.086

 Reply: We are sorry for your experience 😔

Enter review (type 'exit'): exit
```

Type `exit` to end the session. Visualisations are generated automatically afterwards.

---

## Output & Visualisations

After exiting the input loop, the script produces three visualisations:

### A. Sentiment Distribution Pie Chart
Shows the proportion of Positive / Negative / Neutral predictions across all reviews entered during the session.

### B. Per-Sector Word Clouds
A word cloud is generated for each sector that received at least one review during the session, using the raw input text grouped by detected sector.

### C. Combined Word Cloud
All text from `advanced_nlp_dataset.csv` is merged into a single 800 × 400 px word cloud with a white background, giving an at-a-glance view of the most frequent terms across the entire dataset.

---

## Project Structure

```
├── nlp_project_org.py            # Main script
├── advanced_nlp_dataset.csv      # Labelled training dataset (999 rows)
├── expanded_sector_keywords.csv  # Sector keyword dictionary (182 keywords)
└── README.md                     # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | any | Data loading and session tracking |
| `nltk` | any | Stopwords, lemmatizer, WordNet synonyms |
| `scikit-learn` | any | TF-IDF, Logistic Regression, evaluation |
| `wordcloud` | any | Word cloud generation |
| `matplotlib` | any | Chart and word cloud rendering |

---

## How It Works — Quick Summary

```
Customer review (text input)
        │
        ├──► Preprocessing ──► TF-IDF ──► Logistic Regression ──► Sentiment
        │
        └──► Keyword matching against sector dictionary ──► Sector
                                                                │
                                                 Sentiment + Sector
                                                                │
                                                        Auto Reply
```

---

*Built as part of an NLP course project. Dataset is synthetically generated for demonstration purposes.*
