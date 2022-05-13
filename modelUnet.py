from turtle import forward
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 

class DoubleConv(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential (
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    """ This inital sequential network corrresponds to the first 2 layers of network before maxpool downsampling"""

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], ):
        super(UNET, self).__init__()
        # We are also going to add a module list for batchnorm evals on 
        # expanding(up-sampling) and contracting(down-sampling) paths respectively
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Then we proceed to add the maxpool layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # From paper, we will now implement the contractive(down-sampling) path
        # We do this by iterating through features above and selescting correspoding in channel
        # from paper
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature # i.e mapping to 64 from paper

        
        # We do the same for expanding(up-sampling) path

        for feature in reversed(features):
            # Take a look at architecture illustration to understand why we transpose
            self.ups.append(
                nn.ConvTranspose2d(
                # Here we mutliply features by 2 because of skip connection concatenation
                feature*2, feature, kernel_size=2, stride=2,
                )
            
            )
            self.ups.append(DoubleConv(feature*2, feature))
        

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2D(features[0], out_channels, kernel_size=1)

    
    # Let's implement the forward pass for UNET model

    def forward(self, x):
        # store skip connections below

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            X = self.pool(x)
        
        X = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        