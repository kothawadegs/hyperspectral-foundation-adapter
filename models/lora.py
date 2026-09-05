import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Native PyTorch Low-Rank Adaptation (LoRA) layer.
    Factorizes weight update: W_eff = W_frozen + (alpha / r) * (B @ A)
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False
            
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r
        
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original_linear(x)
        delta = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + delta
