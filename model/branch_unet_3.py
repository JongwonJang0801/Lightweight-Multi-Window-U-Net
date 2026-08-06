import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_win=3):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_win, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_win, padding='same'),
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
        
        down_drop = self.dropout(down_st)
        down_sum = self.conv_sum(down_drop)
        down_sum2 = self.dropout(down_sum)
        
        #down_sum = self.conv_sum(down_st)
        #p = self.pool(down_sum)
        p = self.pool(down_sum2)

        return down_sum, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, fea_in, fea_out, back_in):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.conv2 = DoubleConv(fea_in, fea_out)
        self.conv3 = DoubleConv(back_in, fea_out*2)
        self.dropout = nn.Dropout(p=0.2)
        

    def forward(self, x1, x2):
        x3 = x1.permute(0,2,1,3)
        
        #x3 = self.dropout(x3)
        x3 = self.conv2(x3)
        x3 = x3.permute(0,2,1,3)
        
        #x1 = self.up(x1)
        x1 = self.up(x3)
        
        
        x4 = x2.permute(0,2,1,3)
        #x4 = self.dropout(x4)
        x4 = self.conv3(x4)
        x2 = x4.permute(0,2,1,3)
        
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class fea_Dense(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, in_ch_2, out_ch_2, in_ch_3, out_ch_3):
        super().__init__()
        self.dense_1 = nn.Sequential(
            nn.Linear(in_channels_1, out_channels_1),
            nn.ReLU(inplace=True),
        )
        
        self.dense_2 = nn.Sequential(
            nn.Linear(in_ch_2, out_ch_2),
            nn.ReLU(inplace=True),
        )
        
        
        self.dense_3 = nn.Sequential(
            nn.Linear(in_ch_3, out_ch_3),
            nn.ReLU(inplace=True),
        )
        
        self.conv = DoubleConv(in_channels_1, out_channels_1)
        self.conv2 = DoubleConv(in_ch_3, out_ch_3)

    def forward(self, x1, x2):
        #x3 = self.conv(x1)
        
        x3 = x1.permute(0,3,2,1)
        x3 = self.dense_1(x3)
        x3 = x3.permute(0,3,2,1)
        
        x3 = self.dense_2(x3)
        
        x4 = torch.cat([x3,x2], 2)
        
        x4 = x4.permute(0,2,1,3)
        x4 = self.conv2(x4)
        x4 = x4.permute(0,2,1,3)
        
        return x4

class myModel(nn.Module):
    
    def __init__(self, in_channels, fea_len, num_classes, cfg_encoder):
        super().__init__()
        
        self.fea_len = fea_len
        #self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_1 = DownSample(1, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        #self.down_convolution_4 = DownSample(64, 128)

        #self.bottle_neck = DoubleConv(512, 1024)
        self.bottle_neck = DoubleConv(256, 512)
        #self.bottle_neck = DoubleConv(128, 256)
        
        self.dense1 = fea_Dense(1024, 512, 4, 8, fea_len//16+fea_len//8,fea_len//16)
        self.dense2 = fea_Dense(512, 256, 8, 16, fea_len//8+fea_len//4,fea_len//8)
        self.dense3 = fea_Dense(256, 128, 16, 32, fea_len//8+fea_len//2,fea_len//8)
        self.dense4 = fea_Dense(128, 64, 32, 64, fea_len//8+fea_len,fea_len//8)
        
        self.up_convolution_1 = UpSample(1024, 512, fea_len//16, fea_len//16, 6)
        self.up_convolution_2 = UpSample(512, 256, fea_len//8, fea_len//16, fea_len//4)
        self.up_convolution_3 = UpSample(256, 128, fea_len//8, fea_len//16, fea_len//2)
        #self.up_convolution_3 = UpSample(256, 128, fea_len//8, fea_len//16, fea_len//2)
        self.up_convolution_4 = UpSample(128, 64, fea_len//8, fea_len//16, fea_len)
        #self.conv1d = nn.Conv1d(16*self.fea_len//8, num_classes, 1)
        self.conv1d = nn.Conv1d(64*self.fea_len//8, num_classes, 1)
        
        self.out = nn.Conv2d(in_channels=8, out_channels=num_classes, kernel_size=1)
        

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        
        #down_4, p4 = self.down_convolution_4(p3)
        #b = self.bottle_neck(p4)
        b = self.bottle_neck(p3)
        
        #up_1 = self.dense1(b, down_4)
        up_2 = self.dense2(b, down_3)
        up_3 = self.dense3(up_2, down_2)
        up_4 = self.dense4(up_3, down_1)
        
        out = torch.reshape(up_4, (up_4.size()[0],-1,up_4.size()[-1]))
        out =self.conv1d(out)
        return out