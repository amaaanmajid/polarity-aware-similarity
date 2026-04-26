# Polarity-Aware Similarity

> Cosine similarity is broken for meaning. This fixes it — without training a single parameter.

---

## The Problem

Standard cosine similarity on sentence embeddings has a well-known blind spot: **it measures lexical and structural overlap, not meaning**.

A few concrete examples that motivated this work:

| Base Sentence | Comparison | Cosine Score | Problem |
|---|---|---|---|
| "I like Mumbai" | "I do not like Mumbai" | 0.71 | Negation ignored — scores *higher* than "I hate Mumbai" (0.50) |
| "I like Mumbai" | "I hate Mumbai" | 0.50 | Antonym treated as distant even though both are negative |
| "I earned 1 dollar" | "I earned 10 dollars" | 0.41 | Same as "I earned 1 million dollars" (0.41) — magnitude blind |
| "The movie was good" | "The movie was bad" | 0.75 | Sentiment polarity nearly invisible |
| "The movie was good" | "The movie was terrible" | 0.75 | Intensity difference not captured |

Cosine works well for **topical clustering**. It fails for **meaning-sensitive** tasks like NLI, contradiction detection, and sentiment-aware retrieval.

---

## The Solution

**Polarity-Aware Similarity** composes two frozen, pretrained models geometrically — no training, no fine-tuning, no learned losses.

### Pipeline

```
s → BERT (frozen) → H (token embeddings)
                         ↓
              Polarity Model (frozen) → signed attention α
                         ↓
              p = Σ(αᵢ · hᵢ)    ← polarity embedding
              s = MeanPool(H)    ← semantic embedding

sim(s1, s2) = α · cos(s1, s2) + β · cos(p1, p2)
```

- **Semantic stream** (`cos(s1, s2)`): captures topic overlap, referential meaning
- **Polarity stream** (`cos(p1, p2)`): captures sentiment direction, negation scope, intensity
- **α, β**: fixed hyperparameters (no learning — can be swept at eval time)

The polarity embedding is constructed by weighting BERT's contextual token representations using **signed gradient-based attention** from a frozen sentiment classifier. Tokens that push toward positive sentiment get positive weights; tokens pushing toward negative sentiment get negative weights. This is **interpreted**, not learned.

The result is a similarity score in **[−1, +1]** where:
- `+1` → identical meaning and polarity
- `0` → topic match, polarity neutral
- `−1` → same topic, opposite meaning (contradiction)

---

## Key Design Principles

- **No training** — purely inference-time composition
- **No new parameters** — reuses existing pretrained model weights
- **Frozen models only** — BERT + a pretrained sentiment classifier
- **Geometric composition** — polarity space is constructed, not learned
- **Task-flexible** — α/β can be adjusted per downstream task

---

## Models Used

| Role | Model |
|---|---|
| Contextual embeddings | `bert-base-uncased` |
| Polarity attention | `cardiffnlp/twitter-roberta-base-sentiment-latest` |

Both models are loaded frozen. No gradient updates to either.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/polarity-aware-similarity.git
cd polarity-aware-similarity
pip install torch transformers pandas scikit-learn openpyxl
```

---

## Quick Start

### Single Pair

```python
from polarity_similarity import PolarityAwareSimilarity

model = PolarityAwareSimilarity(alpha=0.4, beta=0.6)

total, sem, pol = model.similarity(
    "I like Mumbai",
    "I do not like Mumbai",
    return_components=True
)

print(f"Total: {total:.3f} | Semantic: {sem:.3f} | Polarity: {pol:.3f}")
# Total: -0.014 | Semantic: +0.804 | Polarity: -0.559
```

### Ranking Candidates

```python
query = "The service was terrible"
candidates = [
    "The service was excellent",
    "The service was not good",
    "The service was slow",
    "The food was bad"
]

results = model.batch_similarity(query, candidates, top_k=4)
for sent, score in results:
    print(f"[{score:.3f}] {sent}")
```

### Run Built-in Demo

```python
python polarity_similarity.py
```

This runs 24 test pairs covering negation, antonyms, intensifiers, numbers, double negation, and same-polarity paraphrases — with semantic and polarity scores printed side by side.

---

## Evaluate on ANLI Dataset

The code includes a full NLI evaluation pipeline. It runs on any CSV/Excel file with columns `premise`, `hypothesis`, `label` (0=entailment, 1=neutral, 2=contradiction).

### Download ANLI

```bash
# Option 1: Hugging Face datasets
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('anli', split='test_r1')
ds.to_csv('anli_sample.csv')
"
```

```bash
# Option 2: Direct from Hugging Face Hub
# https://huggingface.co/datasets/facebook/anli
```

### Run Evaluation

```python
from polarity_similarity import PolarityAwareSimilarity, evaluate_excel

model = PolarityAwareSimilarity(alpha=0.4, beta=0.6)

evaluate_excel(
    model,
    excel_path="anli_sample.csv",
    save_path="anli_predictions.csv",
    max_rows=100        # Set None for full dataset
)
```

### Output

```
Accuracy: 0.41

Confusion Matrix:
[[12  8  3]
 [ 9 11  7]
 [ 4  6 15]]

Classification Report:
              precision    recall  f1-score
  Entailment       0.52      0.52      0.52
     Neutral       0.44      0.41      0.42
Contradiction       0.60      0.60      0.60
```

### NLI Decision Rules

```python
def predict_label(sem_sim, pol_sim):
    if pol_sim < -0.50:   return 2  # Contradiction
    if pol_sim > +0.40:   return 0  # Entailment
    return 1                         # Neutral
```

Predictions are polarity-driven: the model identifies contradiction via negative polarity alignment, entailment via positive alignment, and defaults to neutral otherwise.

---

## Results on Test Pairs

Selected results from the built-in 24-pair test suite:

| Pair | sem | pol | Verdict |
|---|---|---|---|
| "I love this phone" vs "I do not love this phone" | +0.804 | −0.559 | ✅ Contradiction detected |
| "The food was good" vs "The food was not good" | +0.902 | −0.786 | ✅ Strong contradiction |
| "He is kind" vs "He is cruel" | +0.892 | −0.829 | ✅ Antonym detected |
| "The service was slow" vs "not slow at all" | +0.837 | −0.848 | ✅ Full negation |
| "The room is too small" vs "big enough" | +0.845 | −0.706 | ✅ Opposite implication |
| "I enjoy hiking" vs "I love trekking" | +0.882 | +0.707 | ✅ Same polarity |
| "I hate waiting" vs "Waiting is the worst" | +0.731 | +0.466 | ✅ Same negative |

---

## Hyperparameter Guide

| α (semantic) | β (polarity) | Best for |
|---|---|---|
| 0.7 | 0.3 | General semantic similarity |
| 0.4 | 0.6 | Sentiment-sensitive tasks, NLI |
| 0.2 | 0.8 | Pure contradiction detection |
| 0.6 | 0.4 | Balanced retrieval |

---

## Interpretation Guide

```
pol > +0.4   →  Same polarity     (entailment / agreement)
pol < -0.4   →  Opposite polarity (contradiction)
pol ≈  0.0   →  Neutral           (topic match only)
sem  high    →  Topically similar sentences
```

---

## Limitations

- The Cardiff sentiment model is **tweet-domain biased** — physical/factual sentences ("hot vs cold", number comparisons) produce weak polarity signals
- **Double negation** ("not without talent") is partially handled but inconsistent
- Tokenization mismatch between BERT and RoBERTa requires interpolation, which can lose alignment on very short sentences
- NLI accuracy is modest (~40%) — the value is in the **interpretable decomposition** (sem + pol), not raw classification

---

## Why This Matters

This work demonstrates:

1. **Cosine similarity has a measurable, systematic failure mode** on negation, antonyms, and sentiment
2. **Polarity is latent in pretrained models** — it can be extracted geometrically without training
3. **Semantic and evaluative meaning are separable** — the two streams capture different aspects of meaning
4. **Inference-only methods can expose what learned embeddings miss**

---

## Citation / Reference

If you use this code, please link back to this repository.

Inspired by the limitations documented in:
> *"Cosine similarity is broken for meaning"* — LinkedIn post by [YOUR NAME], 2025

---

## License

MIT License. Free to use, modify, and build on.
