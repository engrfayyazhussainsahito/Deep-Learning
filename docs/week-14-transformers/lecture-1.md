# Week 14: BERT & The Encoder-only Paradigm

## Theory: Contextualizing Representations

Before BERT, embeddings were static (Word2Vec). BERT (Bidirectional Encoder Representations from Transformers) introduced the idea that a word's meaning is entirely dependent on its surrounding context.

### 1. The Architecture: The Transformer Encoder
BERT consists of a stack of Transformer Encoders. Unlike the original Transformer, it has no decoder. Its primary goal is to produce high-quality, contextualized vectors for every input token.

### 2. Pre-training Objectives
BERT is trained on massive corpora using two unsupervised tasks:

#### Masked Language Modeling (MLM)
Instead of predicting the next word, BERT masks a random 15% of the tokens in a sentence and tries to predict them using context from both the left and right.
$\text{Input: "The [MASK] sat on the mat."} \rightarrow \text{Prediction: "cat"}$

#### Next Sentence Prediction (NSP)
To understand relationships between sentences, BERT is given two sentences (A and B) and must predict if B actually follows A in the original text.

### 3. Downstream Fine-tuning
Once pre-trained, BERT acts as a feature extractor. For a specific task (e.g., sentiment analysis), we add a single linear layer on top of the `[CLS]` token (the special token at the start of every sequence) and train only that layer.

:::tip Academic Note
The `[CLS]` token's final hidden state is viewed as a summary of the entire sequence. This is a remarkably efficient way to perform classification without needing to aggregate (pool) all the other token vectors.
:::

## Modular PyTorch Implementation: BERT-based Feature Extraction

We use the HuggingFace `transformers` library to implement a professional feature extraction pipeline.

```python
from transformers import BertTokenizer, BertModel
import torch

class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract(self, text):
        # Tokenize and convert to tensors
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # last_hidden_state: (batch, seq_len, hidden_dim)
        # We extract the [CLS] token (index 0) for sentence-level representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

# Test
extractor = BERTFeatureExtractor()
texts = ["Deep learning is a subset of AI.", "I love learning about neural networks."]
embeddings = extractor.extract(texts)

# Compute Cosine Similarity
similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print(f"Sentence Similarity: {similarity.item():.4f}")
```

## Rigorous Assignment 14: Encoder Dynamics

1.  **Bidirectionality vs Causality**: Explain why a bidirectional encoder (BERT) cannot be used for text generation (auto-regression).
2.  **The Masking Dilemma**: In MLM, we only mask 15% of tokens. Why not mask 50%? How does this affect the training signal?
3.  **Custom Head Implementation**: Implement a custom `nn.Linear` head on top of BERT's `[CLS]` token to classify a dataset of movie reviews into 5 stars.
