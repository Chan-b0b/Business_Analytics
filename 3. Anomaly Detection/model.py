from torch import nn
import torch
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(encoded_space_dim = latent_dim)
        self.decoder = Decoder(encoded_latent_dim = latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_latent_dim, transpose=True):
        super().__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
        # self.decoder_conv = nn.Sequential(
        #     nn.Upsample(scale_factor = 2, mode='bilinear'),
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Upsample(scale_factor = 2, mode='bilinear'), 
        #     nn.Conv2d(16, 8, 3, padding=2),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.Upsample(scale_factor = 2, mode='bilinear'),           
        #     nn.Conv2d(8, 1, 3, padding=1)
        # )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x