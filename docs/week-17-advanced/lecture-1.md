# Week 17: Reinforcement Learning (RL) Foundations

## Theory: Learning through Interaction

Unlike supervised learning, where we have a ground-truth label for every input, **Reinforcement Learning (RL)** involves an **Agent** interacting with an **Environment** to maximize a cumulative scalar **Reward**.

### 1. The Markov Decision Process (MDP)
The mathematical framework for RL is the MDP, defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
*   **State ($\mathcal{S}$)**: The current configuration of the environment.
*   **Action ($\mathcal{A}$)**: The set of all possible moves the agent can make.
*   **Transition Probability ($\mathcal{P}$)**: $P(s' | s, a)$ — the probability of landing in state $s'$ after taking action $a$ in state $s$.
*   **Reward ($\mathcal{R}$)**: The immediate reward received after a transition.
*   **Discount Factor ($\gamma \in [0, 1]$)**: Determines the importance of future rewards.

### 2. The Bellman Equation
The value of a state $V(s)$ is the expected sum of discounted future rewards. The fundamental Bellman optimality equation is:
$V^*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]$

For action-value functions (Q-values):
$Q^*(s, a) = \mathbb{E} [r + \gamma \max_{a'} Q^*(s', a') | s, a]$

### 3. Deep Q-Networks (DQN)
In complex environments (like Atari), the state space is too large for a table. DQN uses a neural network to approximate $Q(s, a)$.

#### Two Critical Stability Mechanisms
1.  **Experience Replay**: Storing transitions $(s, a, r, s')$ in a buffer and sampling them randomly to break the temporal correlation of training data.
2.  **Target Network**: Using a separate, slowly-updating network to calculate the target value $y = r + \gamma \max Q_{target}(s', a')$, preventing the model from "chasing its own tail."

:::tip Academic Note
RLHF (Reinforcement Learning from Human Feedback), used to align LLMs, is essentially a DQN-like process where the "Reward Function" is replaced by a neural network trained to mimic human preferences.
:::

## Modular PyTorch Implementation: The Q-Network

We implement a professional Q-Network and a basic epsilon-greedy action selection strategy.

```python
import torch
import torch.nn as nn
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def select_action(q_net, state, epsilon):
    """
    Epsilon-Greedy Action Selection
    """
    if random.random() < epsilon:
        return torch.randint(0, 2, (1,)).item() # Explore
    else:
        with torch.no_grad():
            q_values = q_net(state)
            return torch.argmax(q_values).item() # Exploit

# Setup
state_dim, action_dim = 4, 2 # e.g., CartPole environment
q_net = QNetwork(state_dim, action_dim)
state = torch.randn(1, state_dim)

action = select_action(q_net, state, epsilon=0.1)
print(f"Selected Action: {action}")
```

## Rigorous Assignment 17: Convergence & Policy

1.  **Convergence Proof**: Prove that if the learning rate $\alpha$ satisfies the Robbins-Monro conditions ($\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$), then Q-learning converges to the optimal $Q^*$.
2.  **Exploration vs Exploitation**: Compare a fixed $\epsilon$ strategy vs an $\epsilon$-decay strategy. Which one leads to a more stable final policy in a grid-world environment?
3.  **Policy Gradients**: Research the **REINFORCE** algorithm. Explain how it differs from Q-Learning (Value-based) by directly optimizing the policy $\pi(a|s)$.
