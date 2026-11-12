"""SwinIR Implementation

SwinIR (Swin Transformer for Image Restoration) for Super-Resolution.
Based on: https://github.com/JingyunLiang/SwinIR

For SR, the architecture uses:
1. Shallow feature extraction (conv + patch embed)
2. Deep feature extraction: Multiple Residual Swin Transformer Blocks (RSTB) 
   - Each RSTB contains multiple Swin Transformer Blocks with residual connection
   - All RSTBs maintain the same spatial resolution (no downsampling)
3. Reconstruction: Upsampling + output conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Optional, List
from omegaconf import DictConfig
import math


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA)."""

    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Pair-wise relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted windows."""

    def __init__(self, dim: int, input_resolution: Tuple[int, int],
                 num_heads: int, window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # Attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.register_buffer("attn_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: {L} vs {H}*{W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Contains multiple Swin Transformer Blocks with a residual connection.
    Maintains spatial resolution throughout.
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int],
                 depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 norm_layer=nn.LayerNorm, use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # Conv for residual connection
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)

        return x + shortcut


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size: int = 64, patch_size: int = 1,
                 in_chans: int = 4, embed_dim: int = 96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SwinIR(nn.Module):
    """
    SwinIR (Swin Transformer for Image Restoration) for Super-Resolution.

    Args:
        upscale: Upscaling factor (default: 4)
        img_size: Input image size (default: 64)
        window_size: Window size (default: 8)
        depths: Depth of each RSTB (default: [6,6,6,6,6,6])
        embed_dim: Embedding dimension (default: 180)
        num_heads: Number of heads per RSTB (default: [6,6,6,6,6,6])
        mlp_ratio: MLP ratio (default: 2)
        in_chans: Input channels (default: 4 for MS)
        out_chans: Output channels (default: 4 for MS)
    """

    def __init__(self, upscale: int = 4, img_size: int = 64, window_size: int = 8,
                 depths: List[int] = [6, 6, 6, 6, 6, 6],
                 embed_dim: int = 180, num_heads: List[int] = [6, 6, 6, 6, 6, 6],
                 mlp_ratio: float = 2.0, in_chans: int = 4, out_chans: int = 4):
        super().__init__()

        self.upscale = upscale
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = 1
        self.mlp_ratio = mlp_ratio

        # Shallow feature extraction
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size,
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm
        )
        self.pos_drop = nn.Dropout(0.)

        # Deep feature extraction: RSTBs
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,  # Constant dimension for SR
                input_resolution=(img_size, img_size),  # Constant resolution
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0.,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=nn.LayerNorm,
                use_checkpoint=False
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64 * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(64, out_chans, 3, padding=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Shallow feature extraction
        x = self.forward_features(x)

        # Reshape to spatial
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)

        # Upsample
        x = self.upsample(x)

        return x


def create_swinir(config: DictConfig) -> SwinIR:
    """Factory function to create SwinIR from Hydra config."""
    return SwinIR(
        upscale=config.get('upscale', 4),
        img_size=config.get('img_size', 64),
        window_size=config.get('window_size', 8),
        depths=config.get('depths', [6, 6, 6, 6, 6, 6]),
        embed_dim=config.get('embed_dim', 180),
        num_heads=config.get('num_heads', [6, 6, 6, 6, 6, 6]),
        mlp_ratio=config.get('mlp_ratio', 2),
        in_chans=config.get('in_chans', 4),
        out_chans=config.get('out_chans', 4)
    )


# Tiny SwinIR for CPU Development
class TinySwinIR(SwinIR):
    """Tiny SwinIR for CPU-only development (depth=2, smaller embed_dim)."""

    def __init__(self, upscale: int = 4, img_size: int = 64, window_size: int = 8,
                 in_chans: int = 4, out_chans: int = 4):
        super().__init__(
            upscale=upscale,
            img_size=img_size,
            window_size=window_size,
            depths=[2, 2, 2, 2, 2, 2],
            embed_dim=60,
            num_heads=[2, 2, 2, 2, 2, 2],
            mlp_ratio=2,
            in_chans=in_chans,
            out_chans=out_chans
        )


# Main test
if __name__ == "__main__":
    print("Testing SwinIR-base...")

    # Test base model
    model = SwinIR(upscale=4, img_size=64, window_size=8,
                   depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, in_chans=4, out_chans=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Test forward
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        y = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (1, 4, 256, 256)

    # Test tiny model
    tiny = TinySwinIR()
    tiny_params = sum(p.numel() for p in tiny.parameters())
    print(f"Tiny SwinIR parameters: {tiny_params / 1e6:.2f}M")

    with torch.no_grad():
        y_tiny = tiny(x)
    print(f"Tiny output: {y_tiny.shape}")

    print("SwinIR test passed!")