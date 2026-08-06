import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(torch.nn.Module) :
    def __init__(self,input_len,fea_len,n_class,cfg_encoder) :
        self.input_len = input_len
        self.fea_len = fea_len
        super(myModel, self).__init__()
        self.encoder = Encoder(input_len,fea_len,**cfg_encoder)
        self.decoder = Decoder(self.encoder.D_out*fea_len,n_class)
        #self.ctc_layer = torch.nn.Linear(self.encoder.D_out, self.num_label)
        
    def forward(self, x) :
        encode_feature = self.encoder(x)
        #ctc_output = torch.nn.functional.log_softmax(self.ctc_layer(encode_feature) + 1e-6, dim=-1)
        result = self.decoder(encode_feature)
        return result

class Encoder(torch.nn.Module) :
    def __init__(self,input_len, fea_len, n_layers,D_layers,kernel_size) :
        super(Encoder, self).__init__()
        self.input_len=input_len
        self.fea_len=fea_len
        
        module_list = list()
        self.input_dim = 1
        for i in range(n_layers) :
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            # stride=1, padding=0, dilation=1, groups=1, bias=True,
            # padding_mode='zeros', device=None, dtype=None)
            module_list.append(nn.Conv2d(self.input_dim, D_layers[i],
                                         kernel_size=kernel_size[i],
                                         stride=1, padding='same', padding_mode='circular'))
            #module_list.append(torch.nn.BatchNorm2d(D_layers[i]))
            module_list.append(nn.ReLU())
            #module_list.append(nn.Dropout(p=0.2))
            self.input_dim = D_layers[i]
        self.D_out = self.input_dim
        self.layers = nn.ModuleList(module_list)
        
    
    def forward(self,x) :
        for _, layer in enumerate(self.layers) :
            x= layer(x)
        x = x.view(-1, self.D_out*self.fea_len, self.input_len)
        return x

class Decoder(torch.nn.Module) :
    def __init__(self, input_len, label_len) :
        super(Decoder, self).__init__()
        module_list=[]
        module_list.append(nn.Conv1d(input_len, 10, 1))
        module_list.append(nn.ReLU())
        module_list.append(nn.Conv1d(10, label_len, 1))
        module_list.append(nn.Softmax())
        self.layers = nn.ModuleList(module_list) 
        
    def forward(self,x) :
        for _, layer in enumerate(self.layers) :
            x= layer(x)
        return x