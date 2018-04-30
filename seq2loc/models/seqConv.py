import torch
import torch.nn as nn

import pdb


class Conv1dLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding = 1):
        super(Conv1dLayer, self).__init__()

        self.layer = nn.Sequential(
                         nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride=stride, padding = padding),
                         nn.BatchNorm1d(ch_out),
                         nn.ReLU(True)
                        )

    def forward(self, input):
        return self.layer(input)
    
class Residual1dLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(Residual1dLayer, self).__init__()
        
        padding_conv = kernel_size/2 - 1
        self.layer = Conv1dLayer(ch_in, ch_out, kernel_size, 2, padding_conv)
        self.downsize = torch.nn.AvgPool1d(kernel_size = 2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)
        
    def forward(self, input):
        
        out1 = self.layer(input)
        out2 = self.downsize(input)
        
        out = torch.cat((out1, out2), 1)
        
        return out


class SeqConvResidClassifier(nn.Module)    :
    def __init__(self, ch_in, ch_out, kernel_size, layers_deep, ch_intermed, pooling_type = 'max'):
        super(SeqConvResidClassifier, self).__init__()
        
        kernel_size = 16
        stride = 2
        padding = 1
        
        self.blocks = nn.ModuleList([])
        
        for i in range(layers_deep):
            self.blocks.append(Residual1dLayer(ch_in+ch_intermed*i, ch_intermed, kernel_size))
            
        if pooling_type is 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_type is 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError('pooling_type must be "max" or "avg"')
            
        self.out_layer = nn.Linear(ch_in+ch_intermed*layers_deep, ch_out)
        
    def forward(self, x):
        
        for block in self.blocks:

            x = block(x)
        
        x = self.pool(x)
        x = torch.squeeze(x)

        out = self.out_layer(x)
        
        return out
    
class SeqConvClassifier(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SeqConvClassifier, self).__init__()
        
        kernel_size = 16
        stride = 2
        padding = 1
        
        self.layer = nn.Sequential(
                                Conv1dLayer(ch_in, 128, kernel_size, stride, padding),
                                Conv1dLayer(128, 256, kernel_size, stride, padding),
                                Conv1dLayer(256, 512, kernel_size, stride, padding),
                                Conv1dLayer(512, 1024, kernel_size, stride, padding),
                                Conv1dLayer(1024, 1024, kernel_size, stride, padding),                        
                                )
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_layer = nn.Linear(1024, ch_out)
        
    def forward(self, input):
        
        out = self.layer(input)
        out = self.pool(out)
        out = torch.squeeze(out)

        out = self.out_layer(out)
        
        return out