"""
Spatial Projector for Alpha Algorithm.
Downsamples 2D CNN feature grids via 2x2 pixel unshuffle/pooling and maps
them into compact visual tokens via a 2-layer GELU MLP.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialProjector(nn.Module):
    """
    Trainable 2-layer GELU MLP spatial projector with 2x2 spatial downsampling.
    Converts 2D CNN feature maps (B, C, H, W) into compact visual token embeddings (B, N, D_llm).
    """
    def __init__(
        self,
        in_channels: int = 1024,
        llm_dim: int = 1536,
        hidden_dim: int = 2048,
        downsample_mode: str = "pixel_unshuffle",  # "pixel_unshuffle" or "pool"
        downsample_factor: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim
        self.downsample_mode = downsample_mode
        self.downsample_factor = downsample_factor

        if downsample_mode == "pixel_unshuffle":
            # 2x2 pixel unshuffle merges 4 spatial elements into channel dimension
            mlp_in_dim = in_channels * (downsample_factor ** 2)
        elif downsample_mode == "pool":
            mlp_in_dim = in_channels
        else:
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

        self.mlp_in_dim = mlp_in_dim

        # 2-layer GELU MLP projector with LayerNorm for stable multimodal training
        self.norm = nn.LayerNorm(mlp_in_dim)
        self.fc1 = nn.Linear(mlp_in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsamples (B, C, H, W) spatially by downsample_factor.
        """
        b, c, h, w = x.shape
        f = self.downsample_factor

        # Ensure spatial dims are divisible by downsample factor
        pad_h = (f - (h % f)) % f
        pad_w = (f - (w % f)) % f
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            h, w = x.shape[2], x.shape[3]

        if self.downsample_mode == "pixel_unshuffle":
            # Reshape into (B, C, H//2, 2, W//2, 2) and permute to (B, H//2, W//2, C*4)
            h_out, w_out = h // f, w // f
            x = x.view(b, c, h_out, f, w_out, f)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(b, h_out, w_out, c * (f * f))  # (B, H_out, W_out, C*4)
            # Flatten spatial dimensions to token sequence: (B, H_out * W_out, C*4)
            tokens = x.view(b, h_out * w_out, -1)
            return tokens
        else:
            # 2x2 spatial average pooling
            x = F.avg_pool2d(x, kernel_size=f, stride=f)  # (B, C, H_out, W_out)
            x = x.permute(0, 2, 3, 1).contiguous()         # (B, H_out, W_out, C)
            tokens = x.view(b, -1, c)                      # (B, H_out * W_out, C)
            return tokens

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            feature_map: Spatial 2D feature map from ConvNeXt CNN (B, C, H, W).
        Returns:
            Visual tokens of shape (B, num_tokens, llm_dim).
        """
        tokens = self._downsample(feature_map)  # (B, N, mlp_in_dim)
        tokens = self.norm(tokens)
        hidden = self.act(self.fc1(tokens))
        out_tokens = self.fc2(hidden)           # (B, N, llm_dim)
        return out_tokens

    def get_token_count(self, cnn_grid_size: Tuple[int, int]) -> int:
        """Computes the number of compact visual tokens for a given CNN grid size."""
        gh, gw = cnn_grid_size
        f = self.downsample_factor
        return (gh // f) * (gw // f)
