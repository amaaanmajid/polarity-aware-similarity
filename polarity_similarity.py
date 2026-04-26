"""Polarity-Aware Similarity (Exact Implementation)

CRITICAL: No training, no fine-tuning, no learned losses.
Only: reuse existing models + geometric composition.

Pipeline (from your document):
    s → BERT(frozen) → H → Pretrained Polarity Attention(frozen) → α → p
    s → BERT(frozen) → s (semantic)
    
    sim(s1,s2) = α·cos(s1,s2) + β·cos(p1,p2)
"""

import torch
import torch.nn.functional as F
 

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification
)
from typing import Tuple, List, Optional
import numpy as np


class PolarityAwareSimilarity:
    """
    Implementation of polarity-aware similarity using frozen models.
    
    Key principles:
    1. No training - only inference
    2. Reuse pretrained polarity-attention models
    3. Geometric composition of semantic + polarity spaces
    """
    
    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        polarity_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        alpha: float = 0.7,  # Semantic weight
        beta: float = 0.3,   # Polarity weight
        device: str = None
    ):
        """
        Args:
            bert_model: Frozen BERT for contextual embeddings
            polarity_model: Pretrained polarity-attention model
            alpha, beta: Fixed hyperparameters (no learning)
            device: cuda or cpu
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.beta = beta
        
        # Load frozen BERT
        print(f"Loading frozen BERT: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model).to(self.device)
        self.bert.eval()
        
        # Load frozen polarity-attention model
        print(f"Loading frozen polarity model: {polarity_model}")
        self.polarity_tokenizer = AutoTokenizer.from_pretrained(polarity_model)
        self.polarity_model = AutoModelForSequenceClassification.from_pretrained(
            polarity_model,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        self.polarity_model.eval()
    
    def get_contextual_embeddings(self, sentence: str) -> Tuple[torch.Tensor, dict]:
        """
        Step 1: H = BERT(s) [FROZEN]
        
        Returns token embeddings H = [h1, ..., hn], hi ∈ R^d
        No gradients, no adaptation.
        
        Args:
            sentence: Input text
            
        Returns:
            H: [n_tokens, d] contextual embeddings
            inputs: Tokenizer outputs
        """
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # [1, n, d] → [n, d]
        H = outputs.last_hidden_state.squeeze(0)
        
        return H, inputs
    
    def extract_polarity_attention(self, sentence: str, H: torch.Tensor) -> torch.Tensor:
        """
        Step 2: α = PolarityAttention(s) [FROZEN MODEL]
        
        Extract token-level attention weights from pretrained model.
        These weights already encode:
            - Polarity-bearing words
            - Negation scope
            - Intensity modifiers
        
        This attention is INTERPRETED, not learned by us.
        
        Args:
            sentence: Input text
            H: [n, d] BERT embeddings (for alignment)
            
        Returns:
            α: [n] attention weights where αi ≥ 0, Σαi = 1
        """
        inputs = self.polarity_tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.polarity_model(**inputs)
        
        # Extract attention from last layer
        # outputs.attentions: tuple of [1, num_heads, seq_len, seq_len]
        last_layer_attention = outputs.attentions[-1]  # [1, num_heads, seq_len, seq_len]
        
        # Average across heads: [1, num_heads, seq_len, seq_len] → [seq_len, seq_len]
        attention_matrix = last_layer_attention.squeeze(0).mean(dim=0)
        
        # Get attention from [CLS] to all tokens (sentence-level importance)
        # This captures which tokens are important for polarity classification
        cls_attention = attention_matrix[0, :]  # [seq_len]
        
        # Remove special tokens ([CLS], [SEP])
        alpha = cls_attention[1:-1]
        
        # Normalize: Σαi = 1
        alpha = alpha / (alpha.sum() + 1e-8)
        
        # Align with BERT tokenization length
        bert_len = H.shape[0]
        if len(alpha) != bert_len:
            # Interpolate to match token counts
            alpha = F.interpolate(
                alpha.unsqueeze(0).unsqueeze(0),
                size=bert_len,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return alpha
    
    def compute_polarity_embedding(self, H: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Step 3: p = Σ(αi · hi)
        
        Project BERT embeddings using polarity attention.
        
        Key point:
            - BERT provides context
            - Attention provides polarity focus
            - NO new parameters introduced
        
        Args:
            H: [n, d] contextual embeddings
            α: [n] attention weights
            
        Returns:
            p: [d] polarity-aware embedding
        """
        # Ensure dimensions match
        if len(alpha) != H.shape[0]:
            alpha = F.interpolate(
                alpha.unsqueeze(0).unsqueeze(0),
                size=H.shape[0],
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Weighted pooling
        p = (alpha.unsqueeze(1) * H).sum(dim=0)
        
        return p
    
    
    def compute_semantic_embedding(self, H: torch.Tensor) -> torch.Tensor:
        """
        Step 4: s = MeanPool(H) or h[CLS]
        
        Standard semantic embedding - topic-dominant, polarity-agnostic.
        
        Args:
            H: [n, d] contextual embeddings
            
        Returns:
            s: [d] semantic embedding
        """
        # Mean pooling (can also use [CLS] token)
        s = H.mean(dim=0)
        
        return s
    
    def semantic_similarity(self, s1: torch.Tensor, s2: torch.Tensor) -> float:
        """
        Step 5a: sim_sem = cos(s1, s2)
        
        Captures:
            - Topic overlap
            - Referential similarity
        
        Args:
            s1, s2: [d] semantic embeddings
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        return F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
    
    def polarity_similarity(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """
        Step 5b: sim_pol = cos(p1, p2)
        
        Captures:
            - Agreement vs contradiction
            - Negation-induced flips
            - Sentiment alignment
        
        Args:
            p1, p2: [d] polarity embeddings
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        return F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
    def get_signed_polarity_attention(self, sentence: str, H: torch.Tensor):
        """
        Returns signed attention:
        + = pushes positive sentiment
        - = pushes negative sentiment
        Magnitude = importance × strength
        """

        # --- Tokenize for polarity model ---
        inputs = self.polarity_tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- Get embeddings with gradients ON ---
        embeds = self.polarity_model.get_input_embeddings()(inputs["input_ids"])
        embeds = embeds.detach()
        embeds.requires_grad_(True)

        # --- Forward pass ---
        outputs = self.polarity_model(
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )

        # --- Extract attention (importance) ---
        last_attn = outputs.attentions[-1]      # [1, heads, seq, seq]
        attn = last_attn.squeeze(0).mean(0)     # [seq, seq]
        cls_attn = attn[0]                      # [seq]
        alpha = cls_attn[1:-1]                 # drop <s>, </s>
        alpha = alpha / (alpha.sum() + 1e-8)

        # --- Backward pass for gradients ---
        POS_INDEX = 2  # Cardiff: 0=neg,1=neu,2=pos
        pos_logit = outputs.logits[0, POS_INDEX]

        self.polarity_model.zero_grad()
        pos_logit.backward()

        grads = embeds.grad.squeeze(0)
        embeds_flat = embeds.detach().squeeze(0)

        # --- Convert gradient to signed strength ---
        scores = (grads * embeds_flat).sum(dim=1)
        signed_scores = scores[1:-1]

        # --- Combine importance × direction/strength ---
        signed_alpha = alpha * signed_scores
        signed_alpha = signed_alpha / (signed_alpha.abs().sum() + 1e-8)

        # --- Align to BERT token length ---
        bert_len = H.shape[0]
        if len(signed_alpha) != bert_len:
            signed_alpha = F.interpolate(
                signed_alpha.unsqueeze(0).unsqueeze(0),
                size=bert_len,
                mode="linear",
                align_corners=False
            ).squeeze()
        # print("SENTENCE:", sentence)
        # print("Logits:", outputs.logits.detach().cpu().numpy())
        # print("Signed scores:", signed_scores.detach().cpu().numpy())
        # print("Sum signed:", signed_scores.sum().item())

        return signed_alpha

    def similarity(
        self,
        sent1: str,
        sent2: str,
        return_components: bool = False
    ) -> float:
        """
        Step 6: Final polarity-aware similarity
        
        sim(s1, s2) = α·sim_sem + β·sim_pol
        
        Where α, β are fixed hyperparameters (can be swept at eval time).
        
        This makes the method:
            - Task-flexible
            - Fully inference-only
            - Reproducible
        
        Args:
            sent1, sent2: Input sentences
            return_components: If True, return (total, sem_sim, pol_sim)
            
        Returns:
            Combined similarity score
        """
        # 1. Get contextual embeddings H (frozen BERT)
        H1, _ = self.get_contextual_embeddings(sent1)
        H2, _ = self.get_contextual_embeddings(sent2)
        
        # 2. Extract polarity attention α (frozen polarity model)
        alpha1 = self.get_signed_polarity_attention(sent1, H1)
        alpha2 = self.get_signed_polarity_attention(sent2, H2)

        
        # 3. Compute polarity embeddings p
        p1 = self.compute_polarity_embedding(H1, alpha1)
        p2 = self.compute_polarity_embedding(H2, alpha2)
        
        # 4. Compute semantic embeddings s
        s1 = self.compute_semantic_embedding(H1)
        s2 = self.compute_semantic_embedding(H2)
        
        # 5. Compute similarities
        sim_sem = self.semantic_similarity(s1, s2)
        sim_pol = self.polarity_similarity(p1, p2)

        # 6. Combine with fixed hyperparameters
        sim_total = self.alpha * sim_sem + self.beta * sim_pol
        print("DEBUG:",
        "sem =", sim_sem,
        "pol =", sim_pol,
        "alpha =", self.alpha,
        "beta =", self.beta,
        "total =", sim_total)
        
        if return_components:
            return sim_total, sim_sem, sim_pol
        
        return sim_total
    
    def batch_similarity(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates by polarity-aware similarity to query.
        
        Args:
            query: Reference sentence
            candidates: List of candidate sentences
            top_k: Return top k (None = all)
            
        Returns:
            Sorted list of (sentence, score)
        """
        scores = [(cand, self.similarity(query, cand)) for cand in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return scores[:top_k]
        return scores
    

# ============================================================================
# DEMONSTRATION: NEGATION & POLARITY SENSITIVITY
# ============================================================================

def demo():
    """
    Demonstrate that standard cosine fails on negation,
    while polarity-aware similarity succeeds.
    """
    print("=" * 80)
    print("POLARITY-AWARE SIMILARITY (FROZEN MODELS ONLY)")
    print("=" * 80)
    print("\nKey principles:")
    print("  • No training, no fine-tuning, no learned losses")
    print("  • Reuse existing polarity-attention models")
    print("  • Geometric composition + similarity computation")
    print("=" * 80)
    
    # Initialize
    model = PolarityAwareSimilarity(
        alpha=0.6,  # Semantic weight
        beta=0.3    # Polarity weight
    )
    
    # Test case 1: Negation sensitivity
    print("\n[TEST 1] NEGATION SENSITIVITY")
    print("-" * 80)
    
    query = "I like mumbai"
    candidates = [
        "I do not like mumbai",
        "I hate mumbai",           # Similar (same polarity)
        "Delhi is better than mumbai",   # Negated (opposite polarity)
        "Mumbai is good",        # Opposite sentiment
        "I stay in mumbai",          # Paraphrase (same polarity)
        "I hate delhi"              # Strong opposite
    ]
    
    print(f"\nQuery: '{query}'\n")
    
    results = model.batch_similarity(query, candidates)
    for i, (cand, score) in enumerate(results, 1):
        total, sem, pol = model.similarity(query, cand, return_components=True)
        print(f"{i}. [{total:.3f}] {cand}")
        print(f"   ├─ Semantic sim: {sem:.3f} (topic overlap)")
        print(f"   └─ Polarity sim: {pol:.3f} (sentiment alignment)\n")
    
    # Test case 2: Intensity modifiers
    print("\n[TEST 2] INTENSITY MODIFIERS")
    print("-" * 80)
    
    query = "The service was very slow"
    candidates = [
        "The service was extremely slow",
        "The service was slightly slow",
        "The service was not slow",
        "The service was fast"
    ]
    
    print(f"\nQuery: '{query}'\n")
    
    results = model.batch_similarity(query, candidates, top_k=4)
    for i, (cand, score) in enumerate(results, 1):
        total, sem, pol = model.similarity(query, cand, return_components=True)
        print(f"{i}. [{total:.3f}] {cand}")
        print(f"   ├─ Semantic: {sem:.3f}")
        print(f"   └─ Polarity: {pol:.3f}\n")
    
    print("=" * 80)
    print("\nWhy this is legit research (even without training):")
    print("  • Redefines similarity geometry")
    print("  • Separates evaluative vs referential meaning")
    print("  • Exposes latent structure in pretrained models")
    print("  • Demonstrates failure mode of cosine similarity")
    print("=" * 80)
# ==============================
# NLI DECISION RULES
# ==============================

SEM_HIGH = 0.60
POL_NEUTRAL = -0.40


def predict_label(sem_sim, pol_sim):
    """
    Polarity-only decision
    0 = entailment (same polarity)
    1 = neutral (weak / near-zero polarity)
    2 = contradiction (opposite polarity)
    """

    # Strong negative alignment → contradiction
    if pol_sim < -0.50:
        return 2

    # Positive alignment → entailment
    if pol_sim > 0.40:
        return 0

    # Near zero → neutral
    return 1

 # neutral
# ==============================
# EXCEL ACCURACY EVALUATION
# ==============================

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_excel(model, excel_path, save_path=None, max_rows=None):
    """
    Runs polarity-aware NLI on an Excel dataset

    Required columns:
      premise, hypothesis, label

    Optional:
      save_path → save predictions back to Excel
      max_rows → test on subset (for speed)
    """

    if excel_path.lower().endswith(".csv"):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)


    if max_rows:
        df = df.head(max_rows)

    y_true = []
    y_pred = []

    predictions = []

    print("\nRunning evaluation...")
    print("Rows:", len(df))
    print("-" * 50)

    for idx, row in df.iterrows():
        premise = str(row["premise"])
        hypothesis = str(row["hypothesis"])
        true_label = int(row["label"])

        total, sem, pol = model.similarity(
            premise,
            hypothesis,
            return_components=True
        )

        pred_label = predict_label(sem, pol)

        y_true.append(true_label)
        y_pred.append(pred_label)

        predictions.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": true_label,
            "pred_label": pred_label,
            "semantic_sim": sem,
            "polarity_sim": pol,
            "total_score": total
        })

        if idx % 10 == 0:
            print(f"Processed {idx+1} rows")

    # --- Metrics ---
    acc = np.mean(np.array(y_true) == np.array(y_pred))

    print("\nAccuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Entailment", "Neutral", "Contradiction"]
    ))

    # --- Save results ---
    if save_path:
        out_df = pd.DataFrame(predictions)
        if save_path.lower().endswith(".csv"):
            out_df.to_csv(save_path, index=False)
        else:
            out_df.to_excel(save_path, index=False)


    return acc


if __name__ == "__main__":

    model = PolarityAwareSimilarity(alpha=0.4, beta=0.6)

    # Format: (sentence1, sentence2, description)
    test_pairs = [

        # --- NEGATION ---
        ("I love this phone", "I do not love this phone", "Direct negation"),
        ("The food was good", "The food was not good", "Negated adjective"),
        ("She is happy", "She is not happy", "Negated state"),

        # --- ANTONYMS / CONTRAST ---
        ("The movie was amazing", "The movie was terrible", "Strong antonyms"),
        ("He is kind", "He is cruel", "Opposite traits"),
        ("The water is hot", "The water is cold", "Physical opposites"),

        # --- INTENSIFIERS / DEGREE ---
        ("The service was slow", "The service was extremely slow", "Intensifier"),
        ("The service was slow", "The service was slightly slow", "Downscaler"),
        ("The service was slow", "The service was not slow at all", "Full negation"),

        # --- NUMBERS / QUANTITY ---
        ("I waited 5 minutes", "I waited 50 minutes", "Number difference"),
        ("The package arrived in 2 days", "The package arrived in 20 days", "Delivery time"),
        ("He scored 90 marks", "He scored 40 marks", "Score difference"),

        # --- TOO / ENOUGH ---
        ("The coffee is too hot", "The coffee is not hot enough", "Too vs not enough"),
        ("The room is too small", "The room is big enough", "Too vs enough"),
        ("She is too tired to work", "She is energetic enough to work", "Too tired vs energetic"),

        # --- DOUBLE NEGATION ---
        ("It is not uncommon", "It is common", "Double negation = positive"),
        ("He is not without talent", "He has talent", "Double negation paraphrase"),

        # --- CONDITIONAL / SOFTENED ---
        ("I might like it", "I definitely like it", "Certainty difference"),
        ("The food was okay", "The food was incredible", "Mild vs strong positive"),

        # --- SAME POLARITY (should score HIGH) ---
        ("I enjoy hiking", "I love trekking", "Same sentiment, different words"),
        ("This is a great product", "This product is excellent", "Paraphrase - both positive"),
        ("I hate waiting", "Waiting is the worst", "Same negative sentiment"),

        # --- TOPIC SAME, POLARITY OPPOSITE ---
        ("Mumbai is a great city", "Mumbai is a terrible city", "Same topic, opposite sentiment"),
        ("The teacher was helpful", "The teacher was unhelpful", "Negated adjective"),

    ]

    print("=" * 90)
    print(f"{'#':<3} {'Total':>6} {'Sem':>6} {'Pol':>6}  Description + Sentences")
    print("=" * 90)

    for i, (s1, s2, desc) in enumerate(test_pairs, 1):
        total, sem, pol = model.similarity(s1, s2, return_components=True)
        print(f"\n{i:<3} [{total:+.3f}]  sem={sem:+.3f}  pol={pol:+.3f}  → {desc}")
        print(f"     S1: {s1}")
        print(f"     S2: {s2}")

    print("\n" + "=" * 90)
    print("Interpretation guide:")
    print("  pol > +0.4  → same polarity (agreement)")
    print("  pol < -0.4  → opposite polarity (contradiction)")
    print("  pol ≈  0.0  → neutral / topic-only match")
    print("  sem  high   → topically similar sentences")
