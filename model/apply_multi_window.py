import torch.nn as nn
import torch


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

class Conv_1y(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_win=3):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_win, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv_1y(in_channels, out_channels)
        self.conv_5 = Conv_1y(in_channels, out_channels, 5)
        self.conv_7 = Conv_1y(in_channels, out_channels, 7)
        self.conv_sum = Conv_1y(out_channels*3, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        down = self.conv(x)
        down_5 = self.conv_5(x)
        down_7 = self.conv_7(x)
        down_st = torch.cat((down,down_5,down_7), dim=1)
        
        down_sum = self.conv_sum(down_st)
        p = self.pool(down_sum)

        return down_sum, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)
   

class myModel(nn.Module):
    def __init__(self, in_channels, fea_len, num_classes, cfg_encoder):
        super().__init__()
        self.fea_len = fea_len
        self.down_convolution_1 = DownSample(1, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)
        
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        self.conv1d = nn.Conv1d(64*self.fea_len, num_classes, 1)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        b = self.bottle_neck(p4)
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = torch.reshape(up_4, (up_4.size()[0],-1,up_4.size()[-1]))
        out =self.conv1d(out)
        return out