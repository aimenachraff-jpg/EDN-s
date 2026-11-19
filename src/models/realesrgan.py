"""
Real-ESRGAN Implementation

Real-ESRGAN with RRDBNet architecture for perceptual super-resolution.
Protocol: 4 input channels (MS bands), 4 output channels, scale=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from omegaconf import DictConfig


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) for RRDB."""
    
    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x  # Residual scaling


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (RRDB)."""
    
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet architecture for Real-ESRGAN.
    
    Args:
        num_in_ch: Input channels (default: 4 for MS)
        num_out_ch: Output channels (default: 4 for MS)
        num_feat: Feature channels (default: 64)
        num_block: Number of RRDB blocks (default: 23)
        scale: Upscaling factor (default: 4)
    """
    
    def __init__(self, num_in_ch: int = 4, num_out_ch: int = 4,
                 num_feat: int = 64, num_block: int = 23, scale: int = 4):
        super().__init__()
        self.scale = scale
        
        # First convolution
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.Sequential(*[RRDB(num_feat) for _ in range(num_block)])
        
        # After body
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.upsample = nn.Sequential()
        if scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        else:
            raise NotImplementedError(f"Scale {scale} not supported")
        
        # Final convolutions
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize
        for m in [self.conv_first, self.conv_body, self.conv_hr, self.conv_last]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.upsample(feat))
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out


class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style Discriminator for Real-ESRGAN.
    Used during GAN training (not for inference).
    """
    
    def __init__(self, num_in_ch: int = 4, num_feat: int = 64):
        super().__init__()
        
        def block(in_c, out_c, stride, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.net = nn.Sequential(
            # 256x256 -> 128x128
            block(num_in_ch, num_feat, 2, norm=False),
            block(num_feat, num_feat, 1),
            # 128x128 -> 64x64
            block(num_feat, num_feat * 2, 2),
            block(num_feat * 2, num_feat * 2, 1),
            # 64x64 -> 32x32
            block(num_feat * 2, num_feat * 4, 2),
            block(num_feat * 4, num_feat * 4, 1),
            # 32x32 -> 16x16
            block(num_feat * 4, num_feat * 8, 2),
            block(num_feat * 8, num_feat * 8, 1),
            # 16x16 -> 8x8
            block(num_feat * 8, num_feat * 8, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat * 8, 1, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], -1)


class RealESRGAN(nn.Module):
    """
    Real-ESRGAN with RRDBNet generator.
    
    For inference: uses only the generator (RRDBNet).
    For training: uses generator + discriminator + perceptual loss.
    """
    
    def __init__(self, net_g: Dict[str, Any], scale: int = 4):
        super().__init__()
        self.scale = scale
        
        # Generator (RRDBNet)
        self.generator = RRDBNet(
            num_in_ch=net_g.get('num_in_ch', 4),
            num_out_ch=net_g.get('num_out_ch', 4),
            num_feat=net_g.get('num_feat', 64),
            num_block=net_g.get('num_block', 23),
            scale=scale
        )
        
        # Discriminator (not used for inference)
        self.discriminator = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass (generator only)."""
        return self.generator(x)
    
    def setup_training(self, use_discriminator: bool = True):
        """Setup for training (creates discriminator if needed)."""
        if use_discriminator and self.discriminator is None:
            self.discriminator = VGGStyleDiscriminator(
                num_in_ch=self.generator.conv_first.in_channels,
                num_feat=64
            )


def create_realesrgan(config: DictConfig) -> RealESRGAN:
    """Factory function to create Real-ESRGAN from Hydra config."""
    net_g = config.get('net_g', {})
    scale = config.get('scale', 4)
    return RealESRGAN(net_g=net_g, scale=scale)


# =============================================================================
# Pretrained Weight Loading (Real-ESRGAN official weights)
# =============================================================================

class RealESRGANPretrained(RealESRGAN):
    """Real-ESRGAN with support for loading official pretrained weights."""
    
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
        
        # Handle Real-ESRGAN checkpoint format
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        
        # Adapt first conv for 4 channels if pretrained on 3 channels
        if adapt_3ch and self.generator.conv_first.in_channels == 4:
            if 'conv_first.weight' in state_dict:
                w = state_dict['conv_first.weight']  # [64, 3, 3, 3]
                new_w = w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                state_dict['conv_first.weight'] = new_w
        
        # Load generator weights
        gen_state = {k: v for k, v in state_dict.items() if not k.startswith('discriminator')}
        missing, unexpected = self.generator.load_state_dict(gen_state, strict=False)
        if missing:
            print(f"RealESRGAN: Missing generator keys: {missing[:5]}...")
        if unexpected:
            print(f"RealESRGAN: Unexpected generator keys: {unexpected[:5]}...")
        
        print(f"RealESRGAN: Loaded pretrained weights from {path}")


# =============================================================================
# Loss Functions for Real-ESRGAN Training
# =============================================================================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layer_weights: Dict[str, float] = None,
                 vgg_type: str = 'vgg19', use_input_norm: bool = True,
                 range_norm: bool = False, criterion: str = 'l1'):
        super().__init__()
        
        if layer_weights is None:
            layer_weights = {
                '4': 0.1,   # conv1_2 (after ReLU)
                '9': 0.1,   # conv2_2
                '18': 1.0,  # conv3_4
                '27': 1.0,  # conv4_4
                '36': 1.0,  # conv5_4
            }
        
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        # Load VGG
        if vgg_type == 'vgg19':
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        else:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        # Build sequential up to max layer needed
        max_layer = max(int(k) for k in layer_weights.keys())
        self.vgg = nn.Sequential()
        for i in range(max_layer + 1):
            self.vgg.add_module(str(i), vgg[i])
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.vgg.eval()
        
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Normalization
        if use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # x, gt: [B, C, H, W] in [0, 1]
        
        # Adapt for 4-channel input (use first 3 channels or mean)
        if x.shape[1] == 4:
            x_vgg = x[:, :3]  # Use RGB channels
            gt_vgg = gt[:, :3]
        else:
            x_vgg = x
            gt_vgg = gt
        
        # Normalize
        if self.use_input_norm:
            x_vgg = (x_vgg - self.mean) / self.std
            gt_vgg = (gt_vgg - self.mean) / self.std
        
        if self.range_norm:
            x_vgg = (x_vgg + 1) / 2
            gt_vgg = (gt_vgg + 1) / 2
        
        # Extract features
        x_feats = self._extract_features(x_vgg)
        gt_feats = self._extract_features(gt_vgg)
        
        # Compute weighted loss
        loss = 0
        for layer, weight in self.layer_weights.items():
            loss += weight * self.criterion(x_feats[layer], gt_feats[layer])
        
        return loss
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract VGG features at specified layers."""
        feats = {}
        for name, module in self.vgg.named_children():
            x = module(x)
            if name in self.layer_weights:
                feats[name] = x
        return feats


class GANLoss(nn.Module):
    """GAN Loss for Real-ESRGAN (Vanilla GAN / LSGAN / RaGAN)."""
    
    def __init__(self, gan_type: str = 'vanilla', real_label_val: float = 1.0,
                 fake_label_val: float = 0.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
    
    def get_target_label(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Get target tensor."""
        if target_is_real:
            return torch.full_like(input, self.real_label_val)
        else:
            return torch.full_like(input, self.fake_label_val)
    
    def forward(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.get_target_label(input, target_is_real)
        return self.loss(input, target)


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    print("Testing Real-ESRGAN...")
    
    # Create model
    model = RealESRGAN(
        net_g={
            'num_in_ch': 4,
            'num_out_ch': 4,
            'num_feat': 64,
            'num_block': 23
        },
        scale=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Test forward
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (1, 4, 256, 256)
    
    # Test discriminator
    model.setup_training()
    disc_out = model.discriminator(y)
    print(f"Discriminator output: {disc_out.shape}")
    
    print("Real-ESRGAN test passed!")