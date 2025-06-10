import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, Any


class SpatialAttentionModule(nn.Module):
    """Spatial Attention Module for CNN branch"""

    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module for CNN branch"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class ResNetBackbone(nn.Module):
    """ResNet50 backbone with attention mechanisms"""

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        # Load pretrained ResNet50
        import torchvision.models as models
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the final layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Add attention modules
        self.channel_attention = ChannelAttentionModule(2048)
        self.spatial_attention = SpatialAttentionModule()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class SelectiveScanKernel(nn.Module):
    """Selective Scan Kernel for Mamba operations"""

    def __init__(self, d_model, d_state=16, dt_rank=None, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Parameters for selective scan
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state)))
        self.D = nn.Parameter(torch.ones(d_model))

        # Delta initialization
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x, delta, B, C):
        """
        x: (B, L, D)
        delta: (B, L, D)
        B: (B, L, N)
        C: (B, L, N)
        """
        batch, length, dim = x.shape

        # Compute discretized A and B
        dt = F.softplus(self.dt_proj(delta))  # (B, L, D)
        A = -torch.exp(self.A_log.float())  # (D, N)

        # Discretization
        dtA = torch.einsum('bld,dn->bldn', dt, A)
        dtB = torch.einsum('bld,bln->bldn', dt, B)

        # Selective scan
        states = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(length):
            states = states * torch.exp(dtA[:, i]) + dtB[:, i] * x[:, i:i + 1, :].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', states, C[:, i])
            outputs.append(y)

        output = torch.stack(outputs, dim=1)
        output = output + x * self.D

        return output


class MambaBlock(nn.Module):
    """Mamba Block with selective scan mechanism"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = nn.SiLU()
        self.dt_rank = math.ceil(self.d_model / 16)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.selective_scan = SelectiveScanKernel(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        batch, length, dim = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # 1D Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :length]
        x = rearrange(x, 'b d l -> b l d')

        x = self.activation(x)

        # Selective scan parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Selective scan
        y = self.selective_scan(x, dt, B, C)

        # Gate and output projection
        y = y * self.activation(res)
        output = self.out_proj(y)

        return output


class MultiDirectionalScanning(nn.Module):
    """Multi-directional scanning strategies for dermoscopic images"""

    def __init__(self, height=14, width=14):
        super().__init__()
        self.height = height
        self.width = width

    def spiral_scan(self, x):
        """Spiral scanning from center to periphery"""
        B, L, D = x.shape
        H, W = self.height, self.width
        x_2d = x.view(B, H, W, D)

        # Generate spiral indices
        center_h, center_w = H // 2, W // 2
        indices = []

        for r in range(max(H, W)):
            for angle in np.linspace(0, 2 * np.pi, max(8, 2 * r), endpoint=False):
                h = int(center_h + r * np.cos(angle))
                w = int(center_w + r * np.sin(angle))
                if 0 <= h < H and 0 <= w < W:
                    indices.append(h * W + w)

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        # Ensure all positions are covered
        all_indices = set(range(L))
        remaining = sorted(all_indices - seen)
        unique_indices.extend(remaining)

        # Apply scanning order
        spiral_x = x[:, unique_indices, :]
        return spiral_x

    def radial_scan(self, x):
        """Radial scanning from boundary to center"""
        B, L, D = x.shape
        H, W = self.height, self.width
        x_2d = x.view(B, H, W, D)

        center_h, center_w = H // 2, W // 2
        distances = []

        for h in range(H):
            for w in range(W):
                dist = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
                angle = np.arctan2(h - center_h, w - center_w)
                distances.append((dist, angle, h * W + w))

        # Sort by distance (descending) then angle
        distances.sort(key=lambda x: (-x[0], x[1]))
        radial_indices = [idx for _, _, idx in distances]

        radial_x = x[:, radial_indices, :]
        return radial_x

    def boundary_aware_scan(self, x):
        """Boundary-aware scanning focusing on lesion contours"""
        B, L, D = x.shape
        H, W = self.height, self.width

        # Simple boundary priority (edges first)
        boundary_indices = []
        interior_indices = []

        for h in range(H):
            for w in range(W):
                idx = h * W + w
                if h == 0 or h == H - 1 or w == 0 or w == W - 1:
                    boundary_indices.append(idx)
                else:
                    interior_indices.append(idx)

        combined_indices = boundary_indices + interior_indices
        boundary_x = x[:, combined_indices, :]
        return boundary_x

    def raster_scan(self, x):
        """Traditional left-to-right, top-to-bottom scanning"""
        return x  # Already in raster order

    def forward(self, x):
        """Apply all scanning strategies and concatenate"""
        spiral_x = self.spiral_scan(x)
        radial_x = self.radial_scan(x)
        boundary_x = self.boundary_aware_scan(x)
        raster_x = self.raster_scan(x)

        # Concatenate along feature dimension
        multi_scan_x = torch.cat([spiral_x, radial_x, boundary_x, raster_x], dim=-1)
        return multi_scan_x


class VMambaBlock(nn.Module):
    """VMamba block for vision tasks"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mamba = MambaBlock(dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class VMambaBranch(nn.Module):
    """VMamba branch for global context modeling"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_height = self.patch_width = img_size // patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # ABCDE rule feature embedding
        self.abcde_embed = nn.Linear(5, embed_dim)  # 5 ABCDE features

        # Multi-directional scanning
        self.multi_scan = MultiDirectionalScanning(self.patch_height, self.patch_width)

        # VMamba blocks
        self.blocks = nn.ModuleList([
            VMambaBlock(embed_dim * 4)  # 4x due to multi-directional scanning
            for _ in range(depth)
        ])

        # Projection to reduce dimension back
        self.proj = nn.Linear(embed_dim * 4, embed_dim)

        # Norm and head
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, abcde_features=None):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add position embedding
        x = x + self.pos_embed

        # Add ABCDE features if provided
        if abcde_features is not None:
            abcde_embed = self.abcde_embed(abcde_features).unsqueeze(1)
            x = x + abcde_embed

        # Multi-directional scanning
        x = self.multi_scan(x)  # (B, num_patches, embed_dim * 4)

        # VMamba blocks
        for block in self.blocks:
            x = block(x)

        # Project back to original dimension
        x = self.proj(x)  # (B, num_patches, embed_dim)

        # Global pooling
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = self.avgpool(x).flatten(1)  # (B, embed_dim)

        return x


class StateSpaceFusion(nn.Module):
    """Adaptive state space fusion mechanism"""

    def __init__(self, cnn_dim=2048, vmamba_dim=768, fusion_dim=1024):
        super().__init__()

        # Project features to same dimension
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vmamba_proj = nn.Linear(vmamba_dim, fusion_dim)

        # Adaptive weight generation
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

        # Feature enhancement
        self.enhance = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, cnn_features, vmamba_features):
        # Project to same dimension
        cnn_proj = self.cnn_proj(cnn_features)
        vmamba_proj = self.vmamba_proj(vmamba_features)

        # Concatenate for weight generation
        combined = torch.cat([cnn_proj, vmamba_proj], dim=1)
        weights = self.weight_net(combined)  # (B, 2)

        # Adaptive fusion
        alpha1, alpha2 = weights[:, 0:1], weights[:, 1:2]
        fused = alpha1 * cnn_proj + alpha2 * vmamba_proj

        # Feature enhancement
        enhanced = self.enhance(fused)

        return enhanced, (alpha1, alpha2)


class DermaMamba(nn.Module):
    """Complete DermaMamba architecture"""

    def __init__(self, num_classes=9, img_size=224, patch_size=16,
                 embed_dim=768, depth=12, pretrained=True):
        super().__init__()

        # CNN branch
        self.cnn_branch = ResNetBackbone(pretrained=pretrained)

        # VMamba branch
        self.vmamba_branch = VMambaBranch(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_classes=num_classes
        )

        # Fusion mechanism
        self.fusion = StateSpaceFusion(
            cnn_dim=2048,
            vmamba_dim=embed_dim,
            fusion_dim=1024
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Initialize classifier
        nn.init.trunc_normal_(self.classifier[0].weight, std=0.02)
        nn.init.trunc_normal_(self.classifier[3].weight, std=0.02)

    def forward(self, x, abcde_features=None, return_attention=False):
        # CNN branch
        cnn_features = self.cnn_branch(x)

        # VMamba branch
        vmamba_features = self.vmamba_branch(x, abcde_features)

        # Fusion
        fused_features, fusion_weights = self.fusion(cnn_features, vmamba_features)

        # Classification
        logits = self.classifier(fused_features)

        if return_attention:
            return logits, fusion_weights
        else:
            return logits


def create_dermamamba(num_classes=9, **kwargs):
    """Factory function to create DermaMamba model"""
    model = DermaMamba(num_classes=num_classes, **kwargs)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_dermamamba(num_classes=9)

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    abcde_features = torch.randn(batch_size, 5)  # Optional ABCDE features

    # Forward pass
    with torch.no_grad():
        logits = model(x, abcde_features)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test with attention weights
        logits, weights = model(x, abcde_features, return_attention=True)
        alpha1, alpha2 = weights
        print(f"CNN branch weights: {alpha1.mean().item():.3f}")
        print(f"VMamba branch weights: {alpha2.mean().item():.3f}")
