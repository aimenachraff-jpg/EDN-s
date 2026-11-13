"""
Base Super-Resolution Model Interface

Unified interface for all SR models in the project.
All models should inherit from SRModel to ensure consistent API.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod


class SRModel(nn.Module, ABC):
    """
    Base class for all Super-Resolution models.

    Provides a unified interface with:
    - scale: Upscaling factor
    - in_channels: Input channels
    - out_channels: Output channels
    - forward(): Standard forward pass
    - flops(): Computational complexity estimation

    All SR models should inherit from this class.
    """

    def __init__(self, scale: int = 4, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            lr: Low-resolution input [B, C, H, W]

        Returns:
            Super-resolved output [B, C, H*scale, W*scale]
        """
        raise NotImplementedError

    def flops(self, input_shape: Tuple[int, int, int, int]) -> int:
        """
        Estimate FLOPs for given input shape.

        Args:
            input_shape: (B, C, H, W)

        Returns:
            Estimated FLOPs
        """
        # Default: rough estimate based on parameters and input size
        B, C, H, W = input_shape
        total_params = sum(p.numel() for p in self.parameters())
        # Rough FLOPs estimate: params * spatial * scale_factor
        return int(total_params * H * W * self.scale * self.scale)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_millions(self) -> float:
        """Count parameters in millions."""
        return self.count_parameters() / 1e6

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'scale': self.scale,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'total_params': self.count_parameters(),
            'total_params_m': self.count_parameters_millions(),
        }

    def summary(self) -> str:
        """Get a summary string of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  scale={self.scale},\n"
            f"  in_channels={self.in_channels},\n"
            f"  out_channels={self.out_channels},\n"
            f"  params={self.count_parameters_millions():.2f}M\n"
            f")"
        )


def create_sr_model(model_type: str, config: Dict[str, Any]) -> SRModel:
    """
    Factory function to create SR models.

    Args:
        model_type: 'edsr', 'swinir', 'tiny_swinir', 'realesrgan'
        config: Model configuration dict

    Returns:
        SRModel instance
    """
    if model_type == 'edsr':
        from .edsr import EDSR
        return EDSR(
            n_resblocks=config.get('n_resblocks', 16),
            n_feats=config.get('n_feats', 64),
            scale=config.get('scale', 4),
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4),
        )
    elif model_type == 'swinir':
        from .swinir import SwinIR
        return SwinIR(
            upscale=config.get('upscale', 4),
            img_size=config.get('img_size', 64),
            window_size=config.get('window_size', 8),
            depths=config.get('depths', [6] * 6),
            embed_dim=config.get('embed_dim', 180),
            num_heads=config.get('num_heads', [6] * 6),
            mlp_ratio=config.get('mlp_ratio', 2),
            in_chans=config.get('in_channels', 4),
            out_chans=config.get('out_channels', 4),
        )
    elif model_type == 'tiny_swinir':
        from .swinir import TinySwinIR
        return TinySwinIR(
            upscale=config.get('upscale', 4),
            img_size=config.get('img_size', 64),
            in_chans=config.get('in_channels', 4),
            out_chans=config.get('out_channels', 4),
        )
    elif model_type == 'realesrgan':
        from .realesrgan import RealESRGAN
        return RealESRGAN(
            net_g=config.get('net_g', {'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23}),
            scale=config.get('scale', 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
