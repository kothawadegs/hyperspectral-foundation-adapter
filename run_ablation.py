import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. Synthetic Hyperspectral Data Generation & Dataset
# -----------------------------------------------------------------------------
def create_synthetic_hsi_data(h=128, w=128, c=204, num_classes=6, seed=42):
    np.random.seed(seed)
    wavelengths = np.linspace(400, 2500, c)
    endmembers = []
    
    for i in range(num_classes):
        base = 0.3 + 0.1 * np.sin(wavelengths / 300.0 + i)
        dip1 = 0.25 * np.exp(-((wavelengths - (1400 + i * 150)) / 60) ** 2)
        dip2 = 0.30 * np.exp(-((wavelengths - (2150 + i * 50)) / 40) ** 2)
        spec = np.clip(base - dip1 - dip2 + np.random.normal(0, 0.01, c), 0.05, 0.95)
        endmembers.append(spec)
    endmembers = np.array(endmembers)

    gt = np.zeros((h, w), dtype=int)
    grid_h, grid_w = h // 2, w // 3
    cls_idx = 1
    for r in range(2):
        for c_idx in range(3):
            gt[r * grid_h:(r + 1) * grid_h, c_idx * grid_w:(c_idx + 1) * grid_w] = cls_idx
            cls_idx += 1

    cube = np.zeros((h, w, c), dtype=np.float32)
    for r in range(h):
        for col in range(w):
            label = gt[r, col]
            cube[r, col, :] = endmembers[label - 1] + np.random.normal(0, 0.02, c)

    cube_norm = (cube - cube.min()) / (cube.max() - cube.min())
    return cube_norm, gt


class HyperspectralPatchDataset(Dataset):
    def __init__(self, cube, gt, patch_size=14):
        self.patch_size = patch_size
        pad_width = patch_size // 2
        self.padded_cube = np.pad(cube, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode="reflect")
        
        self.samples = []
        h, w = gt.shape
        for r in range(h):
            for c in range(w):
                label = gt[r, c]
                if label > 0:
                    self.samples.append((r, c, label - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r, c, label = self.samples[idx]
        p = self.patch_size
        patch = self.padded_cube[r:r + p, c:c + p, :]
        tensor_patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)
        return tensor_patch, torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# 2. Architecture & LoRA Layer
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False
            
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.original_linear(x) + ((self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling)


def build_model(mode="lora", in_bands=204, num_classes=6, r=8, device="cuda"):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectral_adapter = nn.Sequential(
                nn.Conv2d(in_bands, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 3, kernel_size=1),
                nn.BatchNorm2d(3)
            )
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            embed_dim = self.backbone.config.hidden_size
            self.classifier = nn.Linear(embed_dim, num_classes)

            if mode == "linear_probe":
                for p in self.backbone.parameters():
                    p.requires_grad = False
            elif mode == "lora":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for layer in self.backbone.encoder.layer:
                    layer.attention.attention.query = LoRALinear(layer.attention.attention.query, r=r)
                    layer.attention.attention.value = LoRALinear(layer.attention.attention.value, r=r)
            elif mode == "full_ft":
                for p in self.backbone.parameters():
                    p.requires_grad = True

        def forward(self, x):
            x_proj = self.spectral_adapter(x)
            outputs = self.backbone(pixel_values=x_proj)
            cls_token = outputs.last_hidden_state[:, 0, :]
            return self.classifier(cls_token), cls_token

    return Net().to(device)


# -----------------------------------------------------------------------------
# 3. Ablation Execution Loop
# -----------------------------------------------------------------------------
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing ablation benchmark on device: {device}")

    # Prepare datasets
    cube_norm, gt = create_synthetic_hsi_data()
    dataset = HyperspectralPatchDataset(cube_norm, gt, patch_size=14)
    
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        stratify=[label for _, _, label in dataset.samples], 
        random_state=42
    )
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=64, shuffle=False)

    modes = ["linear_probe", "lora", "full_ft"]
    epochs = 10
    results = {}

    for mode in modes:
        print(f"\n================ Running Ablation: {mode.upper()} ================")
        model = build_model(mode=mode, device=device)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        print(f"Trainable Params: {trainable_p:,} / {total_p:,} ({100 * trainable_p / total_p:.2f}%)")

        lr = 1e-4 if mode == "full_ft" else 1e-3
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "test_acc": []}
        start_time = time.time()

        for ep in range(epochs):
            model.train()
            running_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits, _ = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(y_b)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    logits, _ = model(x_b)
                    correct += (torch.argmax(logits, dim=-1) == y_b).sum().item()
                    total += len(y_b)

            epoch_loss = running_loss / len(train_idx)
            test_acc = (correct / total) * 100
            history["train_loss"].append(epoch_loss)
            history["test_acc"].append(test_acc)
            print(f"Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Test Acc: {test_acc:.2f}%")

        elapsed = time.time() - start_time
        results[mode] = {
            "params": trainable_p,
            "pct_params": 100 * trainable_p / total_p,
            "final_acc": history["test_acc"][-1],
            "time": elapsed,
            "history": history
        }

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Summary Table & Convergence Plot
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Method':<15} | {'Trainable Params':<18} | {'% Params':<10} | {'Test Acc':<10} | {'Time (s)':<8}")
    print("=" * 70)
    for mode, d in results.items():
        print(f"{mode:<15} | {d['params']:<18,d} | {d['pct_params']:<9.2f}% | {d['final_acc']:<9.2f}% | {d['time']:<8.1f}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode, d in results.items():
        axes[0].plot(range(1, epochs + 1), d["history"]["train_loss"], label=mode)
        axes[1].plot(range(1, epochs + 1), d["history"]["test_acc"], label=mode)

    axes[0].set_title("Training Loss Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Test Accuracy Progression")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    output_img = "ablation_results.png"
    plt.savefig(output_img)
    print(f"\nConvergence plot saved to '{output_img}'.")


if __name__ == "__main__":
    run_benchmark()
