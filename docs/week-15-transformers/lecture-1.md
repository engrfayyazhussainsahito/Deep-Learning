# Week 15: GPT & The Decoder-only Paradigm (LLMs)

## Theory: Generative Pre-training and Scaling Laws

While BERT focuses on understanding (encoding), GPT (Generative Pre-trained Transformer) focuses on creation. GPT is a **Decoder-only** architecture, optimized for the task of **Causal Language Modeling**.

### 1. The Causal Architecture
In a decoder-only model, the attention mechanism is modified with a **Causal Mask** (a lower-triangular matrix). This ensures that the prediction for token $t$ only depends on tokens $1 \dots t-1$.
$\text{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T + \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{V}$
where $\mathbf{M}$ is $-\infty$ for all $j > i$.

### 2. Scaling Laws: The Emergence of Intelligence
The transition from GPT-2 to GPT-3 and beyond revealed **Scaling Laws**: as we increase parameters ($\Phi$), dataset size ($D$), and compute ($C$), the loss decreases predictably. More surprisingly, this leads to **Emergent Abilities**: capabilities like few-shot reasoning and coding that appear only after a certain scale threshold.

### 3. Alignment: From Raw LLM to Assistant
A raw LLM is a "document completer." To make it a "helpful assistant," we use **RLHF (Reinforcement Learning from Human Feedback)**:
1.  **SFT (Supervised Fine-Tuning)**: Training on high-quality (Prompt, Response) pairs.
2.  **Reward Modeling**: Training a separate model to rank multiple responses from best to worst.
3.  **PPO (Proximal Policy Optimization)**: Optimizing the LLM to maximize the reward model's score.

:::tip Academic Note
The "Temperature" parameter $\tau$ in sampling controls the entropy of the output. $\tau \rightarrow 0$ results in Greedy Search (deterministic), while $\tau \rightarrow \infty$ results in random sampling.
:::

## Modular PyTorch Implementation: Causal Sampling Loop

We implement a professional sampling loop with **Top-P (Nucleus) Sampling** to balance coherence and creativity.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt, max_len=50, temp=0.7, top_p=0.9):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs, 
                max_length=max_len, 
                do_sample=True, 
                temperature=temp, 
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Test
generator = LLMGenerator()
prompt = "The first law of thermodynamics states that"
print(f"Generated Text: \n{generator.generate(prompt)}")
```

## Rigorous Assignment 15: Generative Dynamics

1.  **KV Caching**: In a decoder-only model, we recompute the $\mathbf{K}$ and $\mathbf{V}$ matrices for all previous tokens at every step. Explain how **KV Caching** reduces this complexity from $O(N^2)$ to $O(N)$.
2.  **Sampling Comparison**: Compare the output of "Greedy Search" vs "Nucleus Sampling" for the same prompt. In what scenarios does Greedy Search lead to repetitive "loops"?
3.  **RLHF Analysis**: Describe the "Reward Hacking" problem in RLHF. How can a model learn to "trick" the reward model without actually providing a better answer?
