# %%
import torch 
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv3DVarConditionalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, conditional_features=3, latent_dim=48):
        super(Conv3DVarConditionalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
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
        
        # Feature processing for variational part
        fc_size = 64 * 12 * 3 * 5
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        
        # VAE specific: mean and log variance
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Decoder input layer (latent + conditional)
        self.fc_decode = nn.Linear(latent_dim + conditional_features, fc_size)
        
        # Decoder layers
        self.upsample1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(0, 0, 0))
        self.upsample1_relu = nn.ReLU()
        
        self.upsample2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(1, 1, 0))
        self.upsample2_relu = nn.ReLU()
        
        self.upsample3 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=(2, 2, 1), padding=1, output_padding=(0, 0, 0))
        self.upsample3_relu = nn.ReLU()
        
        # Final adjustment layer
        self.final_adjustment = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.final_adjustment_relu = nn.ReLU()
        
        # Output layer
        self.output_layer = nn.Conv3d(8, out_channels, kernel_size=1)
        
    def encode(self, x):
        encoded = self.encoder(x)
        batch_size = x.size(0)
        flattened = encoded.view(batch_size, -1)
        h1 = torch.relu(self.fc1(flattened))
        
        # Get latent parameters
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, cond_features, orig_shape):
        # Combine latent vector with conditional features
        combined = torch.cat([z, cond_features], dim=1)
        
        # Decode
        batch_size = z.size(0)
        fc_out = torch.relu(self.fc_decode(combined))
        
        # Reshape back to feature maps
        feature_maps = fc_out.view(batch_size, 64, 12, 3, 5)
        
        # Decode with transposed convolutions
        x = self.upsample1(feature_maps)          # [B, 32, 25, 7, 5]
        x = self.upsample1_relu(x)
        
        x = self.upsample2(x)                     # [B, 16, 51, 15, 5]
        x = self.upsample2_relu(x)
        
        x = self.upsample3(x)                     # [B, 8, 101, 31, 5]
        x = self.upsample3_relu(x)
        
        # Final adjustments
        x = self.final_adjustment(x)              # [B, 8, 101, 31, 5]
        x = self.final_adjustment_relu(x)
        
        # Final output
        x = self.output_layer(x)                  # [B, 1, 101, 31, 5]
        
        # Ensure exact output dimensions
        if x.shape[2:] != orig_shape[2:]:
            x = nn.functional.interpolate(
                x, 
                size=orig_shape[2:],
                mode='trilinear', 
                align_corners=False
            )
        
        return x
    
    def forward(self, x, cond_features):
        # Store original spatial dimensions
        orig_shape = x.shape
        
        # Encode and get latent parameters
        mu, logvar = self.encode(x)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, cond_features, orig_shape)
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples, cond_features, orig_shape):
        # Generate samples from the latent space
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Make sure conditional features match the number of samples
        if cond_features.size(0) != num_samples:
            cond_features = cond_features.repeat(num_samples, 1)
            
        # Decode the random latent vectors
        samples = self.decode(z, cond_features, orig_shape)
        return samples
        
    def count_params(self):
        nparams = 0
        for param in self.parameters():
            nparams += param.numel()
        return nparams


# Loss function for VAE
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (mean squared error for continuous data)
    # You can replace with BCE if your data is binary
    recon_loss = torch.nn.MSELoss()(recon_x, x)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss = reconstruction loss + beta * KL divergence
    return recon_loss + beta * kl_div, recon_loss, kl_div


# # Usage example:
# batch_size = 4
# model = Conv3DVarConditionalAutoencoder(
#     in_channels=3, 
#     out_channels=1, 
#     conditional_features=3,
#     latent_dim=128
# ).to(device)

# input_tensor = torch.randn(batch_size, 3, 101, 31, 5).to(device)
# conditional_features = torch.randn(batch_size, 3).to(device)  # 3 conditional features

# # Forward pass
# output, mu, logvar = model(input_tensor, conditional_features)
# print(f"Output shape: {output.shape}")  # Should be [batch_size, 1, 101, 31, 5]
# target = output.clone()

# # Calculate loss
# loss, recon_loss, kl_loss = vae_loss_function(output, target, mu, logvar, beta=1.0)
# print(f"Total loss: {loss}, Reconstruction loss: {recon_loss}, KL loss: {kl_loss}")

# # Generate new samples
# orig_shape = input_tensor.shape
# samples = model.sample(batch_size, conditional_features, orig_shape)
# print(f"Generated samples shape: {samples.shape}")

# %% 