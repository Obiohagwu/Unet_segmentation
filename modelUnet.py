from operator import concat
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

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    
    # Let's implement the forward pass for UNET model

    def forward(self, x):
        # store skip connections below

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # This illuistrates the up sampling
            # up, then double conv, up, then double conv
            # While also integrating skip connections
            x = self.ups[idx](x) # up sample
            skip_connection = skip_connections[idx//2] # get skip connectin

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Given that the input is ie 161*161, and upon downsamping with 
            # maxpool, we get floor division by 2 = 80*80, upon upsampling
            # we are supposed to concatenate input and output with skip connections
            # we would have to resize output dimension to input to be able to concatenate
            # via skip connections

            concat_skip = torch.cat((skip_connection, x), dim=1) # For concatenation of skip and upsample
            x = self.ups[idx+1](concat_skip) # then run it through doubleConv
        
        return self.final_conv(x)

#print("Testing: Yes, we are good!")        


# Now, lets run some mini tests to see how our model works with arbitrary input
def test():
    x = torch.randn((3,1,160,160)) # batc_size=3, input_channel=1, input_dim=160*160
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape
    print("This test is mainly to make sure that our input and output are\n the same dims, such that the skip cnnection concats are good.")

if __name__ == "__main__":
        test()