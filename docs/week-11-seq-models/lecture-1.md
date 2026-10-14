# Week 11: Word Embeddings & The Geometry of Language

## Theory: Vectorizing Semantics

Language is discrete (words), but neural networks require continuous input. The goal of **Word Embeddings** is to map words into a high-dimensional vector space where semantic similarity corresponds to geometric distance.

### 1. From One-Hot to Dense Embeddings
A one-hot vector is sparse and lacks meaning. A dense embedding $\mathbf{e}_w \in \mathbb{R}^d$ captures the "essence" of a word.
The core hypothesis is the **Distributional Hypothesis**: "Words that occur in the same contexts tend to have similar meanings."

### 2. Word2Vec: CBOW and Skip-gram
Word2Vec trains embeddings by solving a proxy task:
*   **Continuous Bag of Words (CBOW)**: Predict the target word given the surrounding context words.
*   **Skip-gram**: Predict the surrounding context words given a single target word.

#### The Objective Function
Using the softmax function to predict the word $w_o$ given $w_i$:
$P(w_o | w_i) = \frac{\exp(\mathbf{v}_{w_o}^T \mathbf{v}_{w_i})}{\sum_{w=1}^V \exp(\mathbf{v}_w^T \mathbf{v}_{w_i})}$
To make this efficient for large vocabularies $V$, we use **Negative Sampling**, approximating the sum by sampling a few "negative" (incorrect) words.

### 3. The Geometry of Meaning
In a well-trained embedding space, semantic relationships are represented as linear offsets:
$\text{vec}(\text{"King"}) - \text{vec}(\text{"Man"}) + \text{vec}(\text{"Woman"}) \approx \text{vec}(\text{"Queen"})$

:::tip Academic Note
Modern embeddings like **FastText** improve on Word2Vec by treating words as bags of character n-grams, allowing the model to handle "out-of-vocabulary" words by aggregating the vectors of their sub-components.
:::

## Modular PyTorch Implementation: Semantic Similarity Engine

We implement a tool to calculate the cosine similarity between words and perform vector arithmetic.

```python
import torch
import torch.nn.functional as F

class EmbeddingEngine:
    def __init__(self, vocab_size, embed_dim):
        self.embeddings = torch.randn(vocab_size, embed_dim)
        
    def get_vector(self, word_idx):
        return self.embeddings[word_idx]
    
    def cosine_similarity(self, vec1, vec2):
        # cos(theta) = (A . B) / (||A|| ||B||)
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))

# Mock Vocabulary
vocab = {"king": 0, "man": 1, "woman": 2, "queen": 3, "apple": 4}
engine = EmbeddingEngine(len(vocab), 50)

# Simulate a trained relationship: Queen = King - Man + Woman
v_king = engine.get_vector(vocab["king"])
v_man = engine.get_vector(vocab["man"])
v_woman = engine.get_vector(vocab["woman"])
v_queen_target = v_king - v_man + v_woman

# Compare the result to the actual 'queen' vector
v_queen_actual = engine.get_vector(vocab["queen"])
sim = engine.cosine_similarity(v_queen_target, v_queen_actual)

print(f"Cosine Similarity: {sim.item():.4f}")
```

## Rigorous Assignment 11: Vector Spaces

1.  **Softmax Bottleneck**: Explain why the standard softmax function becomes computationally prohibitive as the vocabulary size $V$ grows to $10^6$. How does Hierarchical Softmax solve this?
2.  **T-SNE Visualization**: Using the `sklearn.manifold.TSNE` tool, project a set of 100 word embeddings from a pre-trained model (like GloVe) into 2D. Identify clusters of similar words.
3.  **Contextual vs Static**: Contrast Word2Vec (static) with BERT (contextual). Explain why the word "bank" has different vectors in "river bank" vs "investment bank" in BERT, but the same vector in Word2Vec.
