import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        
        # GroupNorm + SiLU
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.act = nn.SiLU() # SiLU
        
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return self.nin_shortcut(x) + h


class Upsample(nn.Module):
    """
    Resize + Conv Instead of TransposeConv
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 2 times upsampling
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for decoder"""
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        w = torch.bmm(q, k)  # B, HW, HW
        w = w * (int(C) ** (-0.5))
        w = F.softmax(w, dim=2)

        # Attend to values
        v = v.reshape(B, C, H * W)  # B, C, HW
        w = w.permute(0, 2, 1)  # B, HW, HW
        h_ = torch.bmm(v, w)  # B, C, HW
        h_ = h_.reshape(B, C, H, W)

        h_ = self.proj_out(h_)

        return x + h_


class VQGAN(nn.Module):
    """
    VQGAN-style Decoder with configurable architecture
    
    Args:
        in_channel: Input image channels (default: 3)
        emb_dim: Embedding dimension from VLM tokens (default: 2048)
        channel: Base channel width (default: 512)
        channel_mult: Channel multipliers for each resolution (default: [1, 1, 2, 4])
        n_res_block: Number of ResNet blocks per resolution (default: 2)
        resolution: Target output resolution (default: 224)
        num_groups: Number of groups for GroupNorm (default: 32)
        use_attention: Whether to use attention in mid blocks (default: True)
        attention_resolutions: Resolutions to add attention at (default: [16])
        quantize: Dummy parameter for compatibility with VQVAE (not used)
        n_embed: Dummy parameter for compatibility
        n_res_channel: Dummy parameter for compatibility
        decay: Dummy parameter for compatibility
    """
    
    def __init__(
        self,
        in_channel=3,
        emb_dim=2048,
        channel=256,
        channel_mult=[1, 1, 2, 2, 4],
        n_res_block=2,
        resolution=256,
        num_groups=32,
        use_attention=True,
        grid_size=None,
        input_norm=True,
        **kwargs
    ):
        super().__init__()
        
        if isinstance(resolution, (list, tuple)):
            if len(resolution) != 2:
                raise ValueError(
                    f"resolution must be an int or [height, width], got {resolution}"
                )
            self.output_size = tuple(map(int, resolution))
        else:
            size = int(resolution)
            self.output_size = (size, size)
        self.resolution = resolution
        self.grid_size = tuple(map(int, grid_size)) if grid_size is not None else None
        
        # 1. Initial convolution
        # reversed_mult[0] = 4, start_ch = 256 * 4 = 1024
        ch_mult_reversed = list(reversed(channel_mult))
        start_ch = channel * ch_mult_reversed[0] 
        
        self.input_norm = nn.LayerNorm(emb_dim) if input_norm else nn.Identity()
        self.conv_in = nn.Conv2d(emb_dim, start_ch, kernel_size=3, stride=1, padding=1)
        
        # 2. Middle blocks
        self.mid_blocks = nn.ModuleList()
        self.mid_blocks.append(ResnetBlock(start_ch, start_ch, num_groups))
        if use_attention:
            self.mid_blocks.append(AttentionBlock(start_ch, num_groups))
        self.mid_blocks.append(ResnetBlock(start_ch, start_ch, num_groups))
        
        # 3. Upsampling blocks
        self.up_blocks = nn.ModuleList()
        curr_ch = start_ch
        
        for i in range(len(ch_mult_reversed) - 1):
            block = nn.ModuleList()
            
            # Level 0 (16x16): 1024 -> 512 (mult 4 -> 2)
            out_ch = channel * ch_mult_reversed[i+1]
            
            for _ in range(n_res_block):
                block.append(ResnetBlock(curr_ch, out_ch, num_groups))
                curr_ch = out_ch
            
            # (16->32, 32->64, 64->128, 128->256)
            block.append(Upsample(curr_ch))
            self.up_blocks.append(block)
            
        # 4. Output layer
        self.norm_out = nn.GroupNorm(num_groups=num_groups, num_channels=curr_ch, eps=1e-6, affine=True)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(curr_ch, in_channel, kernel_size=3, padding=1)
        
        print(f"VQGAN Decoder initialized:")
        print(f"  Input: {emb_dim}D embeddings")
        print(f"  Base channels: {channel}")
        print(f"  Channel multipliers: {channel_mult}")
        print(f"  ResNet blocks per level: {n_res_block}")
        print(f"  Target resolution: {self.output_size[0]}x{self.output_size[1]}")
        print(f"  Use attention: {use_attention}")
        print(f"  Input layer norm: {input_norm}")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: VLM tokens of shape (B, num_patches, emb_dim) or (B, emb_dim, H, W)
            
        Returns:
            decoded_images: (B, in_channel, output_height, output_width)
            diff: Dummy tensor for compatibility (always 0)
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Input: (B, num_patches, emb_dim)
            B, L, C = x.shape
            if self.grid_size is None:
                H = W = int(L ** 0.5)
                if H * W != L:
                    raise ValueError(
                        f"Cannot infer a square grid for {L} tokens; pass grid_size explicitly"
                    )
            else:
                H, W = self.grid_size
                if H * W != L:
                    raise ValueError(
                        f"grid_size={self.grid_size} expects {H * W} tokens, got {L}"
                    )
            x = self.input_norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        elif x.dim() == 4:
            # Input already in (B, C, H, W) format
            x = x.permute(0, 2, 3, 1)
            x = self.input_norm(x)
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x.float()
        # Initial conv
        h = self.conv_in(x)
        
        # Middle blocks
        for block in self.mid_blocks:
            h = block(h)
        
        # Upsampling blocks
        for block_list in self.up_blocks:
            for layer in block_list:
                h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        h = torch.tanh(h)
        
        # Optionally resize to exact target resolution if needed
        if h.shape[-2:] != self.output_size:
            h = F.interpolate(h, size=self.output_size, mode='bilinear', align_corners=False)

        # Return dummy diff for compatibility with VQVAE interface
        diff = torch.zeros(1, device=h.device, dtype=h.dtype)
        
        return h, diff
    
    def decode(self, x):
        """Alias for forward, for compatibility"""
        return self.forward(x)[0]
