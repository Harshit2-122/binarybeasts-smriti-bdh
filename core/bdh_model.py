"""
bdh_model.py — Smriti BDH Wrapper
Simplified BDH-style model with Hebbian σ matrix memory.
For hackathon demo: captures the core architectural claim
(constant-size memory that persists across sessions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HebbianMemory(nn.Module):
    """
    σ matrix — the heart of BDH's persistent memory.
    Size stays constant regardless of how many tokens are processed.
    This is the architectural difference from transformers.
    """

    def __init__(self, input_dim: int = 128, memory_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim

        # σ matrix — fixed size, no matter how long the context
        self.sigma = nn.Parameter(
            torch.zeros(input_dim, memory_dim), requires_grad=False
        )
        self.learning_rate = 0.01

    def hebbian_update(self, x: torch.Tensor):
        """
        Update σ matrix via Hebbian rule: Δσ = η · xᵀ · activation(x·σ)
        No backpropagation. No gradient. Pure synaptic strengthening.
        """
        with torch.no_grad():
            # Project input through current memory
            activation = torch.tanh(x @ self.sigma)  # [batch, memory_dim]

            # Hebbian update: strengthen connections that co-activate
            delta = self.learning_rate * (x.T @ activation)  # [input_dim, memory_dim]
            self.sigma.data += delta

            # Normalize to prevent unbounded growth
            self.sigma.data = F.normalize(self.sigma.data, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read from memory — what does σ know about this input?"""
        self.hebbian_update(x)
        return torch.tanh(x @ self.sigma)

    def get_sparse_activation(self, x: torch.Tensor, top_k_percent: float = 0.05):
        """
        Return top ~5% activated synapses (BDH paper claim: sparse activation).
        Used for Experiment 3 — interpretable synapse audit.
        """
        with torch.no_grad():
            activation = torch.tanh(x @ self.sigma)
            flat = activation.abs().flatten()
            k = max(1, int(len(flat) * top_k_percent))
            threshold = torch.topk(flat, k).values.min()
            sparse = activation * (activation.abs() >= threshold).float()
            sparsity = (sparse != 0).float().mean().item()
        return sparse, sparsity


class SmritiBDH(nn.Module):
    """
    Smriti's BDH-inspired model.
    Encodes text → updates σ matrix → generates response context.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, memory_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Simple token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Hebbian memory — the core BDH component
        self.memory = HebbianMemory(embed_dim, memory_dim)

        # Output projection
        self.output_proj = nn.Linear(memory_dim, embed_dim)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode input tokens into embedding space"""
        embedded = self.embedding(token_ids)  # [seq_len, embed_dim]
        # Mean pool for sequence representation
        return embedded.mean(dim=0, keepdim=True)  # [1, embed_dim]

    def forward(self, token_ids: torch.Tensor):
        """Full forward pass — encode → update memory → read memory"""
        x = self.encode(token_ids)
        memory_output = self.memory(x)
        return self.output_proj(memory_output)

    def get_sigma(self) -> torch.Tensor:
        """Return current σ matrix — for saving to disk"""
        return self.memory.sigma.data.clone()

    def set_sigma(self, sigma: torch.Tensor):
        """Load σ matrix from disk — restoring patient memory"""
        self.memory.sigma.data = sigma.clone()


def create_model() -> SmritiBDH:
    """Create a fresh BDH model instance"""
    model = SmritiBDH(vocab_size=1000, embed_dim=128, memory_dim=256)
    model.eval()
    return model
