"""
ConvNeXt CNN Vision Encoder for Alpha Algorithm.
Preserves exact 2D spatial geometry of high-resolution desktop GUI screenshots.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ConvNeXtBlock(nn.Module):
    """
    Standard ConvNeXt block implementation with 7x7 depthwise conv,
    LayerNorm, 1x1 pointwise convs, and GELU activation.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return shortcut + x


class NativeConvNeXtBackbone(nn.Module):
    """
    Native PyTorch fallback for ConvNeXt-Base when timm or internet access
    is unavailable. Matches ConvNeXt-Base architecture: [3, 3, 27, 3] depths
    and [128, 256, 512, 1024] channel dimensions (~88M parameters).
    """
    def __init__(
        self,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (3, 3, 27, 3),
        dims: Tuple[int, ...] = (128, 256, 512, 1024)
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem: 4x4 patch downsampling
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.GroupNorm(1, dims[-1], eps=1e-6)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = self.norm(x)
        return x


class GUIVisionEncoder(nn.Module):
    """
    Hierarchical ConvNeXt CNN encoder for GUI vision-action grounding.
    Downsamples resolution by 32x while retaining exact 2D spatial arrangement
    of UI elements without 1D token serialization distortion.
    """
    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = False,
        in_channels: int = 3,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_timm = False

        if HAS_TIMM:
            try:
                # Attempt to initialize via timm
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=[-1]
                )
                self.use_timm = True
                # Query feature dim from timm feature_info
                self.out_channels = self.backbone.feature_info.channels()[-1]
            except Exception:
                self.backbone = NativeConvNeXtBackbone(in_chans=in_channels)
                self.out_channels = self.backbone.out_dim
        else:
            self.backbone = NativeConvNeXtBackbone(in_chans=in_channels)
            self.out_channels = self.backbone.out_dim

        # ImageNet normalization constants as buffers
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            pixel_values: Float tensor of shape (B, 3, H, W) normalized to [0, 1].
        Returns:
            Spatial feature grid tensor of shape (B, C, H // 32, W // 32).
        """
        # Ensure normalization
        x = (pixel_values - self.mean) / self.std

        if self.use_timm:
            features = self.backbone(x)
            if isinstance(features, (list, tuple)):
                out = features[-1]
            else:
                out = features
        else:
            out = self.backbone(x)

        return out

    def get_output_dim(self) -> int:
        return self.out_channels

    def get_grid_size(self, image_size: int = 768) -> Tuple[int, int]:
        """Calculates 2D feature grid dimensions after 32x downsampling."""
        return (image_size // 32, image_size // 32)
