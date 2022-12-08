import torch
import torch.nn as nn
from config import Config
from crfseg import CRF

config = Config()

class ConvModule(nn.Module):
    def __init__(self, nin, nout):
        super(ConvModule, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.InstanceNorm2d = nn.InstanceNorm2d(nout)
        self.BatchNorm2d = nn.BatchNorm2d(nout)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(config.drop)
        self.depthwise2 = nn.Conv2d(nout, nout, kernel_size=3, padding=1, groups=nout)
        self.pointwise2 = nn.Conv2d(nout, nout, kernel_size=1)
    
    def forward(self,x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.InstanceNorm2d(out)
        out = self.BatchNorm2d(out)
        out = self.ReLU(out)
        out = self.Dropout(out)
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.InstanceNorm2d(out)
        out = self.BatchNorm2d(out)
        out = self.ReLU(out)
        out = self.Dropout(out)

        return out

##################################################################################################
#                                           UNET                                                 #         
##################################################################################################

class U_Net(nn.Module):
    def __init__(self, input_channels=3,
    encoder=[64, 128, 256, 512], decoder=[1024, 512, 256], output_channels=config.k):
        super(U_Net, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        self.first_module = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(2, 2)

        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder])
        
        decoder_out_sizes = [int(x/2) for x in decoder]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]) 
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, output_channels, 1), # No padding on pointwise
            nn.ReLU(),
        ]

        self.last_module = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations

##################################################################################################
#                                           WNET                                                #         
##################################################################################################

class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = U_Net(input_channels=3, encoder=config.encoderLayerSizes,
                                    decoder=config.decoderLayerSizes, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = U_Net(input_channels=config.k, encoder=config.encoderLayerSizes,
                                    decoder=config.decoderLayerSizes, output_channels=3)
        #self.sigmoid = nn.Sigmoid()
        self.crf = CRF(n_spatial_dims=2)

    def forward_encoder(self, x):
        x_res = self.U_encoder(x)
        segmentations = self.softmax(x_res)
        return segmentations
    
    def forward_enc_crf(self, x):
        segmentations = self.forward_encoder(x)
        crf_seg = self.crf(segmentations)
        x_prime       = self.forward_decoder(segmentations)
        return segmentations, crf_seg, x_prime

    def forward_decoder(self, segmentations):
        x_res = self.U_decoder(segmentations)
        reconstructions = x_res 
        return reconstructions

    def forward(self, x): # x is (3 channels 224x224)
        segmentations = self.forward_encoder(x)
        reconstruction       = self.forward_decoder(segmentations)
        return segmentations, reconstruction