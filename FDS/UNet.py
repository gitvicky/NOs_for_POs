# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=5, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Modified number of features to work with smaller input size
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
        # Save original shape for resizing
        self.original_shape = None

    def forward(self, x):
        # Save original shape for resizing at the end
        x = x[...,0]
        self.original_shape = x.shape
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Ensure the output has the exact same spatial dimensions as input
        if logits.shape != self.original_shape:
            logits = F.interpolate(logits, size=(self.original_shape[2], self.original_shape[3]), 
                                  mode='bilinear', align_corners=True)
        
        logits  = logits.unsqueeze(-1)
        return logits

# # Example usage
# if __name__ == "__main__":
#     # Create a sample input
#     batch_size = 4
#     channels = 5
#     height = 101
#     width = 31
#     x = torch.randn(batch_size, channels, height, width)
    
#     # Initialize the model
#     model = UNet(n_channels=channels, n_classes=channels)
    
#     # Forward pass
#     output = model(x)
    
#     # Check output shape
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     assert output.shape == x.shape, "Output shape doesn't match input shape"

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import UNO

class UNOModel(nn.Module):
    """UNO-based model that replaces U-Net.
    
    Takes input of shape [batch_size, 5, 101, 31] and returns output of the same shape.
    """
    def __init__(self, in_channels=5, out_channels=5, n_layers=5):
        super(UNOModel, self).__init__()
        
        # Define the UNO architecture
        
        # Hidden channels for the initial layer
        hidden_channels = 32
        
        # Output channels for each layer in UNO
        uno_out_channels = [32, 32, 32, 32, 32]
        
        # Number of Fourier modes to use in each layer
        uno_n_modes = [[12, 8], [12, 8], [12, 8], [12, 8], [12, 8]]
        
        # Scaling factors for each layer - for downsampling and upsampling
        # First two layers downscale, last two layers upscale, middle layer keeps resolution
        uno_scalings = [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]
        
        # Horizontal skip connections from encoder to decoder
        horizontal_skips_map = {4: 0, 3: 1}
        
        # Create the UNO model
        self.uno = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            lifting_channels=hidden_channels * 2,
            projection_channels=hidden_channels * 2,
            n_layers=n_layers,
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            horizontal_skips_map=horizontal_skips_map,
            domain_padding=0.1,  # Padding hlps with boundary effects
            domain_padding_mode="one-sided",
            fno_skip="linear",
            channel_mlp_skip="linear",
            factorization="tucker",  # Use tensor factorization for efficiency
            rank=0.5  # Tensor factorization rank
        )
    
    def forward(self, x):
        """Forward pass of the UNO model.
        
        Args:
            x: Input tensor of shape [batch_size, 5, 101, 31]
            
        Returns:
            Output tensor of shape [batch_size, 5, 101, 31]
        """
        x = x[...,0] #Only take the first time step
        x = self.uno(x)
        x = x.unsqueeze(-1) # Add a new dimension for time
        return x

# # Example usage
# if __name__ == "__main__":
#     # Create a sample input tensor
#     batch_size = 4
#     channels = 5
#     height = 101
#     width = 31
#     x = torch.randn(batch_size, channels, height, width, 1)
    
#     # Initialize the UNO model
#     model = UNOModel()
    
#     # Forward pass
#     output = model(x)
    
#     # Check output shape
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     assert output.shape == x.shape, "Output shape doesn't match input shape"
# %%
