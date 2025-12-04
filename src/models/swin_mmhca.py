import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .swin_transformer_v2 import SwinTransformer
from .mhca import MHCA
from .common import default_conv

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd d_model_dim.")
        self.d_model = d_model
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.act = nn.ReLU(True)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.conv(x)
        x = self.act(x)
        return x

class SwinMMHCA(nn.Module):
    def __init__(self, n_inputs=1, n_feats=64, scale=4, height=64, width=64):
        super(SwinMMHCA, self).__init__()
        
        self.n_inputs = n_inputs
        
        self.cnn_encoders = nn.ModuleList([
            nn.Sequential(
                default_conv(1, n_feats, 3),
                nn.ReLU(True)
            ) for _ in range(n_inputs)
        ])
        
        self.pos_encoder = PositionalEncoding2D(d_model=n_feats * n_inputs, height=height, width=width)
        
        self.transformer_encoder = SwinTransformer(
            img_size=height,
            patch_size=4,
            in_chans=n_feats * n_inputs,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4
        )
        
        self.mmhca = MHCA(n_feats=192, ratio=4)

        decoder_layers = []
        n_feats_dec = 192
        # Dynamically create decoder based on scale factor
        num_upsample_blocks = int(math.log2(8 * scale))
        for i in range(num_upsample_blocks):
            decoder_layers.append(UpsampleBlock(n_feats_dec, n_feats_dec))
        decoder_layers.append(default_conv(n_feats_dec, 1, 3))
        self.cnn_decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if isinstance(x, list):
            # Multi-input case
            encoded_features = [encoder(xi) for xi, encoder in zip(x, self.cnn_encoders)]
            x = torch.cat(encoded_features, dim=1)
        else:
            # Single-input case
            x = self.cnn_encoders[0](x)
            
        x = self.pos_encoder(x)
        
        features = self.transformer_encoder.forward_features(x)
        
        x_p0 = features[0] 
        
        B, L, C = x_p0.shape
        H = W = int(L**0.5)
        x = x_p0.view(B, H, W, C).permute(0, 3, 1, 2)
        
        x = self.mmhca(x)
        x = self.cnn_decoder(x)
        return x

if __name__ == '__main__':
    # Example of how to use the model
    # Single-input case
    print("Testing single-input case...")
    model_single = SwinMMHCA(n_inputs=1)
    input_single = torch.randn(1, 1, 64, 64)
    output_single = model_single(input_single)
    print(f"Output shape (single-input): {output_single.shape}")
    
    # Multi-input case
    print("\nTesting multi-input case...")
    model_multi = SwinMMHCA(n_inputs=3)
    input1 = torch.randn(1, 1, 64, 64)
    input2 = torch.randn(1, 1, 64, 64)
    input3 = torch.randn(1, 1, 64, 64)
    output_multi = model_multi([input1, input2, input3])
    print(f"Output shape (multi-input): {output_multi.shape}")
