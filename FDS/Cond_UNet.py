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

class ConditionalBlock(nn.Module):
    """Process and inject conditional information"""
    def __init__(self, cond_channels, out_channels):
        super().__init__()
        # MLP to process conditional input
        self.mlp = nn.Sequential(
            nn.Linear(cond_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels * 2)
        )
        
    def forward(self, x, cond):
        # Process conditional input to get scale and shift parameters
        # for feature-wise linear modulation (FiLM)
        batch_size = x.shape[0]
        params = self.mlp(cond)
        scale, shift = params.chunk(2, dim=1)
        
        # Reshape for broadcasting
        scale = scale.view(batch_size, -1, 1, 1)
        shift = shift.view(batch_size, -1, 1, 1)
        
        # Apply FiLM conditioning: x = scale * x + shift
        return scale * x + shift

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=5, cond_channels=3, bilinear=False):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.cond_channels = cond_channels
        self.bilinear = bilinear

        # Modified number of features to work with smaller input size
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        
        # Conditional blocks for each level of the UNet
        self.cond_inc = ConditionalBlock(cond_channels, 32)
        self.cond_down1 = ConditionalBlock(cond_channels, 64)
        self.cond_down2 = ConditionalBlock(cond_channels, 128)
        self.cond_down3 = ConditionalBlock(cond_channels, 256)
        self.cond_down4 = ConditionalBlock(cond_channels, 512 // factor)
        
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        
        # Conditional blocks for upsampling path
        self.cond_up1 = ConditionalBlock(cond_channels, 256 // factor)
        self.cond_up2 = ConditionalBlock(cond_channels, 128 // factor)
        self.cond_up3 = ConditionalBlock(cond_channels, 64 // factor)
        self.cond_up4 = ConditionalBlock(cond_channels, 32)
        
        self.outc = OutConv(32, n_classes)
        
        # Save original shape for resizing
        self.original_shape = None

    def forward(self, x, cond):
        # Save original shape for resizing at the end
        self.original_shape = x.shape
        
        # Encoder path with conditioning
        x1 = self.inc(x)
        x1 = self.cond_inc(x1, cond)
        
        x2 = self.down1(x1)
        x2 = self.cond_down1(x2, cond)
        
        x3 = self.down2(x2)
        x3 = self.cond_down2(x3, cond)
        
        x4 = self.down3(x3)
        x4 = self.cond_down3(x4, cond)
        
        x5 = self.down4(x4)
        x5 = self.cond_down4(x5, cond)
        
        # Decoder path with conditioning
        x = self.up1(x5, x4)
        x = self.cond_up1(x, cond)
        
        x = self.up2(x, x3)
        x = self.cond_up2(x, cond)
        
        x = self.up3(x, x2)
        x = self.cond_up3(x, cond)
        
        x = self.up4(x, x1)
        x = self.cond_up4(x, cond)
        
        logits = self.outc(x)
        
        # Ensure the output has the exact same spatial dimensions as input
        if logits.shape != self.original_shape:
            logits = F.interpolate(logits, size=(self.original_shape[2], self.original_shape[3]), 
                                  mode='bilinear', align_corners=True)
        
        return logits

# # Example usage
# if __name__ == "__main__":
#     # Create a sample input
#     batch_size = 4
#     channels = 5
#     height = 101
#     width = 31
#     x = torch.randn(batch_size, channels, height, width)
    
#     # Create conditional input
#     cond = torch.randn(batch_size, 3)
    
#     # Initialize the model
#     model = ConditionalUNet(n_channels=channels, n_classes=channels, cond_channels=3)
    
#     # Forward pass
#     output = model(x, cond)
    
#     # Check output shape
#     print(f"Input shape: {x.shape}")
#     print(f"Conditional input shape: {cond.shape}")
#     print(f"Output shape: {output.shape}")
#     assert output.shape == x.shape, "Output shape doesn't match input shape"