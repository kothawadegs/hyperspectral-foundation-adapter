import torch
import torch.nn as nn
from transformers import AutoModel
from .lora import LoRALinear

class HyperDINOv2LoRA(nn.Module):
    """
    Hyperspectral Vision Transformer combining a 1D Spectral Projection Adapter 
    with a frozen DINOv2 backbone and LoRA query/value attention updates.
    """
    def __init__(self, in_bands: int = 204, num_classes: int = 6, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        # 1D Spectral Projection Adapter (cross-band reduction)
        self.spectral_adapter = nn.Sequential(
            nn.Conv2d(in_bands, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.BatchNorm2d(3)
        )
        
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # Inject LoRA into Q and V projections across all transformer encoder blocks
        for layer in self.backbone.encoder.layer:
            layer.attention.attention.query = LoRALinear(layer.attention.attention.query, r=lora_r, lora_alpha=lora_alpha)
            layer.attention.attention.value = LoRALinear(layer.attention.attention.value, r=lora_r, lora_alpha=lora_alpha)
            
        embed_dim = self.backbone.config.hidden_size # 384
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x_proj = self.spectral_adapter(x)
        outputs = self.backbone(pixel_values=x_proj)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token), cls_token
