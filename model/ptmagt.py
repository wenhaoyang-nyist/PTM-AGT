"""Model code for PTM-AGT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class GEGLU(nn.Module):
    """GELU(x W_gate) * (x W_val)."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out, bias=False)
        self.v = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.w(x)) * self.v(x)


class AdaptiveLayerNorm(nn.Module):
    """H + alpha * LN(H), with a learnable alpha."""

    def __init__(self, dim: int, init_scale: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.residual_scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 3
        h = x.squeeze(1) if squeeze else x
        out = h + self.residual_scale * self.norm(h)
        return out.unsqueeze(1) if squeeze else out


class GraphTransformerLayer(nn.Module):
    """Pre-normalised attention + gated feed-forward layer."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        hidden_dim = int(8 * dim / 3)
        self.ffn = nn.Sequential(
            GEGLU(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.norm1 = AdaptiveLayerNorm(dim)
        self.norm2 = AdaptiveLayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: True where the two patients are not connected by an edge
        x_in = x.unsqueeze(1)  # [nodes, 1, dim], as expected by MultiheadAttention
        x_norm = self.norm1(x_in)
        attn_out, _ = self.multihead_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x_in = x_in + self.dropout1(attn_out)

        x_out = x_in + self.ffn(self.norm2(x_in))
        return x_out.squeeze(1)


class PTMAGT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 100,
        num_layers: int = 3,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            GraphTransformerLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.layer_attention = nn.Parameter(torch.FloatTensor([1.0 / num_layers] * num_layers))

        self.classifier = nn.Sequential(nn.Linear(hidden_dim, num_classes))
        self.classifier.apply(xavier_init)

        self.set_residual_scale(residual_scale)

    def set_residual_scale(self, value: float) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, AdaptiveLayerNorm):
                    module.residual_scale.data.fill_(value)

    def layer_weights(self) -> torch.Tensor:
        return F.softmax(self.layer_attention, dim=0)

    def encode(self, x: torch.Tensor, attn_mask: torch.Tensor):
        h = self.input_proj(x)
        layer_outputs = []
        for layer in self.layers:
            h = layer(h, attn_mask)
            layer_outputs.append(h)

        weights = self.layer_weights()
        fused = sum(layer_outputs[i] * weights[i] for i in range(self.num_layers))
        return layer_outputs, fused

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        _, fused = self.encode(x, attn_mask)
        return self.classifier(fused)
