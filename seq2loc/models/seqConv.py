import torch
import torch.nn as nn

import pdb

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    
    def forward(self, input):
        return input

class Conv1dLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding = 1, dropout = 0.2):
        super(Conv1dLayer, self).__init__()

        self.layer = nn.Sequential()
        
        if dropout > 0:
                self.layer.add_module('dr', nn.Dropout(dropout))
        
        self.layer.add_module('conv1d', nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride=stride, padding = padding, bias=False))
        self.layer.add_module('bn1d', nn.BatchNorm1d(ch_out))
        self.layer.add_module('relu', nn.ReLU(True))

    def forward(self, input):
        return self.layer(input)
    
class Residual1dLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride = 2, dropout = 0.2, resid_type = 'cat'):
        super(Residual1dLayer, self).__init__()
        
        padding = kernel_size/2 
        self.layer = nn.Sequential(nn.Conv1d(ch_in, ch_in, kernel_size = kernel_size, stride=stride, padding = padding, bias=False),
                                   nn.ReLU(True), 
                                   nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride=1, padding = padding, bias=False))
        
        if stride == 1:
            self.transfer = IdentityLayer()
        else:
            self.transfer = torch.nn.AvgPool1d(kernel_size = stride, stride=stride, padding=0, ceil_mode=True, count_include_pad=False)
        
        self.resid_type = resid_type
        
    def forward(self, input):
        
        out1 = self.layer(input)
        out2 = self.transfer(input)
        
        if self.resid_type == 'cat':
            out = torch.cat((out1, out2), 1)
        elif self.resid_type == 'sum':
            
            try:
                out = out1+out2
            except:
                pdb.set_trace()
        
        return out


class SeqConvResidClassifier(nn.Module)    :
    def __init__(self, ch_in, ch_out, kernel_size, layers_deep, ch_intermed, dropout = 0.2, pooling_type = 'max', resid_type = 'cat', downsize_on_nth = 1):
        super(SeqConvResidClassifier, self).__init__()
        
        
        self.blocks = nn.ModuleList([])
        
        if resid_type == 'sum':
            self.blocks.append(Conv1dLayer(ch_in, ch_intermed, kernel_size, stride = 2, dropout = dropout))
        
        
        for i in range(0, layers_deep):

            if resid_type == 'cat':
                layers_in = ch_in+ch_intermed*(i)
            elif resid_type == 'sum':
                layers_in = ch_intermed
                
            stride = 1
            
            if i%downsize_on_nth == 0:
                stride = 2
            
            self.blocks.append(Residual1dLayer(layers_in, ch_intermed, kernel_size, stride = stride, dropout = dropout, resid_type = resid_type))
            
        if pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError('pooling_type must be "max" or "avg"')
            
        if resid_type == 'cat':
            layers_in = ch_in+ch_intermed*layers_deep
        elif resid_type == 'sum':
            layers_in = ch_intermed
            
        self.out_layer = nn.Linear(layers_in, ch_out)
        
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)

        x = self.pool(x)
        x = torch.squeeze(x)

        out = self.out_layer(x)
        
        return out
    
class SeqConvResidClassifier2(nn.Module)    :
    def __init__(self, ch_in, ch_out, kernel_size, layers_deep, ch_intermed, dropout = 0.2, pooling_type = 'max'):
        super(SeqConvResidClassifier2, self).__init__()
        
        
        self.blocks = nn.ModuleList([])
        

        self.blocks.append(Conv1dLayer(ch_in, ch_intermed, kernel_size, stride = 2, dropout = dropout))
        
        
        for i in range(0, layers_deep):
            
            stride = 1
            
            if i%5 == 0:
                stride = 2
            
            self.blocks.append(Residual1dLayer(ch_intermed, ch_intermed, kernel_size, stride = stride, dropout = dropout, resid_type = 'sum'))
            
        if pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError('pooling_type must be "max" or "avg"')
            
            
        self.out_layer = nn.Linear(ch_intermed, ch_out)
        
        
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