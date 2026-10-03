import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class myModel(nn.Module):
    def __init__(self, in_channels, fea_len, num_classes, cfg_encoder):
        super().__init__()
        self.fea_len = fea_len
        self.encoder_conv_1 = DoubleConv(1,16)
        self.encoder_conv_2 = DoubleConv(16,32)
        self.encoder_conv_3 = DoubleConv(32,64)
        self.encoder_conv_4 = DoubleConv(64,128)

        self.bottle_neck = DoubleConv(128, 256)
        
        self.decoder_conv_1 = DoubleConv(256, 128)
        self.decoder_conv_2 = DoubleConv(128, 64)
        self.decoder_conv_3 = DoubleConv(64, 32)
        self.decoder_conv_4 = DoubleConv(32, 16)
        
        self.conv1d = nn.Conv1d(16*self.fea_len, num_classes, 1)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        

    def forward(self, x):
        p1 = self.encoder_conv_1(x)
        p2 = self.encoder_conv_2(p1)
        p3 = self.encoder_conv_3(p2)
        p4 = self.encoder_conv_4(p3)
        b = self.bottle_neck(p4)
        de1 = self.decoder_conv_1(b)
        de2 = self.decoder_conv_2(de1)
        de3 = self.decoder_conv_3(de2)
        de4 = self.decoder_conv_4(de3)

        out = torch.reshape(de4, (de4.size()[0],-1,de4.size()[-1]))
        out =self.conv1d(out)
        return out