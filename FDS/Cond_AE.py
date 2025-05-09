# %% 
import torch 
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv3DAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, conditional_features=3):
        super(Conv3DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0),  # [B, 16, 50, 15, 5]
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0),  # [B, 32, 25, 7, 5]
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)   # [B, 64, 12, 3, 5]
        )
        
        # Feature integration
        fc_size = 64 * 12 * 3 * 5
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc_cond = nn.Linear(128 + conditional_features, fc_size)
        
        # Decoder with precise output size control
        self.upsample1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(1, 1, 0))
        self.upsample1_relu = nn.ReLU()
        
        self.upsample2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(1, 1, 0))
        self.upsample2_relu = nn.ReLU()
        
        self.upsample3 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(1, 1, 0))
        self.upsample3_relu = nn.ReLU()
        
        # Final adjustment layer to get exact dimensions
        self.final_adjustment = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.final_adjustment_relu = nn.ReLU()
        
        # Output layer
        self.output_layer = nn.Conv3d(8, out_channels, kernel_size=1)

        # #Using linear layers to get the final output size
        # self.final_x = nn.Linear(103,101)
        # self.final_y = nn.Linear(31,31)
        # self.final_z = nn.Linear(8,5)

        
    def forward(self, x, cond_features):
        # Store original spatial dimensions
        _, _, orig_depth, orig_height, orig_width = x.shape
        
        # Encode
        encoded = self.encoder(x)
        
        # Feature integration
        batch_size = x.size(0)
        flattened = encoded.view(batch_size, -1)
        fc1_out = torch.relu(self.fc1(flattened))
        fc2_out = torch.relu(self.fc2(fc1_out))

        combined = torch.cat([fc2_out, cond_features], dim=1)
        fc2_out = torch.relu(self.fc_cond(combined))
        
        # Reshape back to feature maps
        feature_maps = fc2_out.view(batch_size, 64, 12, 3, 5)
        
        # Decode with more precise control
        x = self.upsample1(feature_maps)          # [B, 32, 25, 7, 5]
        x = self.upsample1_relu(x)
        
        x = self.upsample2(x)                     # [B, 16, 51, 15, 5]
        x = self.upsample2_relu(x)
        
        x = self.upsample3(x)                     # [B, 8, 101, 31, 5]
        x = self.upsample3_relu(x)
        
        # Final adjustments to get exact output size
        x = self.final_adjustment(x)              # [B, 8, 101, 31, 5]
        x = self.final_adjustment_relu(x)
        
        # Final output
        x = self.output_layer(x)                  # [B, 1, 101, 31, 5]
        
        # If dimensions are still not exact, use interpolation as a fallback
        if x.shape[2:] != (orig_depth, orig_height, orig_width):
            x = nn.functional.interpolate(
                x, 
                size=(orig_depth, orig_height, orig_width),
                mode='trilinear', 
                align_corners=False
            )

        # # # Final adjustments to get exact output size
        # x = self.final_x(x.permute(0, 1, 3, 4, 2))
        # x = self.final_y(x.permute(0, 1, 4, 3, 2))
        # x = self.final_z(x.permute(0, 1, 2, 4, 3))
        
        return x
    
    def count_params(self):
        nparams = 0
        for param in self.parameters():
            nparams += param.numel()
        return nparams

# # Usage example:
# batch_size = 4
# model = Conv3DAutoencoder(in_channels=3, out_channels=1, conditional_features=3).to(device)
# input_tensor = torch.randn(batch_size, 3, 101, 31, 5).to(device)
# conditional_features = torch.randn(batch_size, 3).to(device)  # 3 conditional features
# output = model(input_tensor, conditional_features)
# print(output.shape)  # Should be [batch_size, 1, 101, 31, 5]

# %% 