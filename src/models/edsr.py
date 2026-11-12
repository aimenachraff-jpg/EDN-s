"""
EDSR-small Implementation

EDSR (Enhanced Deep Residual Networks) for Super-Resolution.
Small variant: 16 residual blocks, 64 features.
Protocol: 4 input channels (MS bands), 4 output channels, scale=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from omegaconf import DictConfig


class ResBlock(nn.Module):
    """Residual block for EDSR."""
    
    def __init__(self, n_feats: int, kernel_size: int = 3, 
                 bias: bool = True, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampler for EDSR."""
    
    def __init__(self, scale: int, n_feats: int, bias: bool = True):
        modules = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(scale.bit_length() - 1)):
                modules.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError(f"Scale {scale} not supported")
        super().__init__(*modules)


class EDSR(nn.Module):
    """
    EDSR (Enhanced Deep Residual Networks) for Super-Resolution.
    
    Args:
        n_resblocks: Number of residual blocks (default: 16 for small)
        n_feats: Number of feature channels (default: 64)
        scale: Upscaling factor (default: 4)
        in_channels: Input channels (default: 4 for MS)
        out_channels: Output channels (default: 4 for MS)
        res_scale: Residual scaling factor (default: 1.0)
        rgb_range: Input range (default: 1.0 for [0,1] normalized)
    """
    
    def __init__(self, n_resblocks: int = 16, n_feats: int = 64,
                 scale: int = 4, in_channels: int = 4, out_channels: int = 4,
                 res_scale: float = 1.0, rgb_range: float = 1.0):
        super().__init__()
        
        self.scale = scale
        self.rgb_range = rgb_range
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Head: initial feature extraction
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        
        # Body: residual blocks
        body = [
            ResBlock(n_feats, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)
        
        # Tail: upsampling + output
        self.upsample = Upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)
        
        # Mean shift for normalization (optional)
        self.sub_mean = nn.Conv2d(in_channels, in_channels, 1)
        self.add_mean = nn.Conv2d(out_channels, out_channels, 1)
        
        # Initialize as identity
        nn.init.eye_(self.sub_mean.weight.view(in_channels, in_channels))
        nn.init.zeros_(self.sub_mean.bias)
        nn.init.eye_(self.add_mean.weight.view(out_channels, out_channels))
        nn.init.zeros_(self.add_mean.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0, rgb_range]
        
        # Optional mean shift
        x = self.sub_mean(x)
        
        # Head
        x = self.head(x)
        res = x
        
        # Body
        res = self.body(res)
        res += x  # Global residual
        
        # Tail
        x = self.upsample(res)
        x = self.tail(x)
        
        # Optional mean shift
        x = self.add_mean(x)
        
        return x


def create_edsr(config: DictConfig) -> EDSR:
    """Factory function to create EDSR from Hydra config."""
    return EDSR(
        n_resblocks=config.get('n_resblocks', 16),
        n_feats=config.get('n_feats', 64),
        scale=config.get('scale', 4),
        in_channels=config.get('in_channels', 4),
        out_channels=config.get('out_channels', 4),
        res_scale=config.get('res_scale', 1.0),
        rgb_range=config.get('rgb_range', 1.0)
    )


# =============================================================================
# EDSR with Pretrained Weight Loading
# =============================================================================

class EDSRPretrained(EDSR):
    """EDSR with support for loading pretrained weights from BasicSR."""
    
    def __init__(self, *args, pretrained_path: Optional[str] = None, 
                 pretrained_3ch: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_path = pretrained_path
        self.pretrained_3ch = pretrained_3ch
        
        if pretrained_path:
            self.load_pretrained(pretrained_path, pretrained_3ch)
    
    def load_pretrained(self, path: str, adapt_3ch: bool = False):
        """Load pretrained weights, adapting for 4-channel input if needed."""
        state_dict = torch.load(path, map_location='cpu')
        
        # Handle BasicSR checkpoint format
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Adapt first conv for 4 channels if pretrained on 3 channels
        if adapt_3ch and self.in_channels == 4:
            if 'head.weight' in state_dict:
                head_w = state_dict['head.weight']  # [64, 3, 3, 3]
                # Average RGB channels and repeat for 4th channel
                new_head = head_w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                state_dict['head.weight'] = new_head
            
            # Similarly for sub_mean if present
            if 'sub_mean.weight' in state_dict:
                sm_w = state_dict['sub_mean.weight']  # [3, 3, 1, 1]
                new_sm = torch.eye(4).view(4, 4, 1, 1)
                new_sm[:3, :3] = sm_w
                state_dict['sub_mean.weight'] = new_sm
                if 'sub_mean.bias' in state_dict:
                    new_bias = torch.zeros(4)
                    new_bias[:3] = state_dict['sub_mean.bias']
                    state_dict['sub_mean.bias'] = new_bias
        
        # Load with strict=False to handle size mismatches
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"EDSR: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"EDSR: Unexpected keys: {unexpected[:5]}...")
        
        print(f"EDSR: Loaded pretrained weights from {path}")


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    print("Testing EDSR-small...")
    
    # Create model
    model = EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4, out_channels=4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output: [1, 4, 256, 256]")
    assert y.shape == (1, 4, 256, 256), f"Shape mismatch: {y.shape}"
    
    print("EDSR-small test passed!")