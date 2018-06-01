import torch
import torch.nn as nn
import numpy as np

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
        
        self.layer.add_module('bn1d', nn.BatchNorm1d(ch_in))
        self.layer.add_module('relu', nn.ReLU(True))
        
        if dropout > 0:
            self.layer.add_module('dr', nn.Dropout(dropout))
        
        self.layer.add_module('conv1d', nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride=stride, padding = padding, bias=False))
        
    def forward(self, input):
        return self.layer(input)
    
    
    
class Residual1dLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride = 2, dropout = 0.2, resid_type = 'cat'):
        super(Residual1dLayer, self).__init__()
        
        padding = kernel_size/2 
        self.layer = nn.Sequential()
        
        if dropout > 0:
            self.layer.add_module('dr', nn.Dropout(dropout))
        
        self.layer.add_module('conv1', nn.Conv1d(ch_in, ch_in, kernel_size = kernel_size, stride=stride, padding = padding, bias=False))
        self.layer.add_module('relu', nn.ReLU(True))
        self.layer.add_module('conv2', nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride=1, padding = padding, bias=False))
        
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
            out = out1+out2

        return out

class Residual2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride = 2, dropout = 0.2, resid_type = 'cat'):
        super(Residual2d, self).__init__()
        
        padding = kernel_size/2 
        self.layer = nn.Sequential()
        
        if dropout > 0:
            self.layer.add_module('dr', nn.Dropout(dropout))
            
        # self.layer.add_module('bn1', nn.BatchNorm2d(ch_in))
        # self.layer.add_module('relu1', nn.ReLU(inplace=True))
        self.layer.add_module('conv1', nn.Conv2d(ch_in, ch_in, kernel_size = 1, stride=1, padding = 0, bias=False))
        
        self.layer.add_module('relu2', nn.ReLU(inplace=True))
        self.layer.add_module('bn2', nn.BatchNorm2d(ch_in))
        self.layer.add_module('conv2', nn.Conv2d(ch_in, ch_out, kernel_size = 4, stride=stride, padding = 1, bias=False))
        
        if stride == 1:
            self.transfer = IdentityLayer()
        else:
            self.transfer = torch.nn.AvgPool2d(kernel_size = stride, stride=stride, padding=0, ceil_mode=True, count_include_pad=False)
        
        self.resid_type = resid_type
        
    def forward(self, input):
        
        out1 = self.layer(input)
        out2 = self.transfer(input)
        
        # pdb.set_trace()
        if self.resid_type == 'cat':
            out = torch.cat((out1, out2), 1)
        elif self.resid_type == 'sum':
            out = out1+out2

        return out
    
# class ResidualTranspose2d(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size, stride = 2, dropout = 0.2, resid_type = 'cat'):
#         super(Residual2d, self).__init__()
        
#         padding = kernel_size/2 
#         self.layer = nn.Sequential()
        
#         if dropout > 0:
#             self.layer.add_module('dr', nn.Dropout(dropout))
        
#         self.layer.add_module('conv1', nn.Conv2d(ch_in, ch_in, kernel_size = 1, stride=1, padding = 1, bias=False))
#         self.layer.add_module('relu', nn.ReLU(True))
#         self.layer.add_module('conv2', nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 4, stride=stride, padding = padding, bias=False))
        
#         if stride == 1:
#             self.transfer = IdentityLayer()
#         else:
#             self.transfer = nn.Upsample(size=None, scale_factor=stride, mode='nearest')
        
#         self.resid_type = resid_type
        
#     def forward(self, input):
        
#         out1 = self.layer(input)
#         out2 = self.transfer(input)
        
#         if self.resid_type == 'cat':
#             out = torch.cat((out1, out2), 1)
#         elif self.resid_type == 'sum':
#             out = out1+out2

#         return out   
    
    
    
class SeqConvResidClassifier(nn.Module):
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

class DenseLayer1d(nn.Module):
    def __init__(self, ch_in, growth_rate, bottleneck = 4, dropout = 0):
        super(DenseLayer1d, self).__init__()
        
        self.layer = nn.Sequential()

        if bottleneck > 0:
            bottleneck_out = bottleneck*growth_rate
            self.layer.add_module('conv1', Conv1dLayer(ch_in, bottleneck_out, 1, stride = 1, padding = 0, dropout = dropout))
            ch_in = bottleneck_out

        self.layer.add_module('conv3', Conv1dLayer(ch_in, growth_rate, 3, stride=1, padding = 1, dropout = dropout))

        self.transfer = IdentityLayer()
    
    def forward(self, input):

        out1 = self.layer(input)
        out2 = self.transfer(input)

        out = torch.cat((out1, out2), 1)

        return out


class DenseBlock1d(nn.Module):
    def __init__(self, ch_in, nblocks, growth_rate = 32, bottleneck = False, dropout = 0):
        super(DenseBlock1d, self).__init__()
        
        self.layer = nn.Sequential()
        
        for i in range(nblocks):
            self.layer.add_module(str(i), DenseLayer1d(ch_in+(i*growth_rate), growth_rate, bottleneck, dropout))
    
    def forward(self, x):
        
        return self.layer(x)
    
class DenseTransfer1d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DenseTransfer1d, self).__init__()
    
        self.layer = nn.Sequential()
        
        self.layer.add_module('bn1d', nn.BatchNorm1d(ch_in))
        self.layer.add_module('conv1d', nn.Conv1d(ch_in, ch_out, kernel_size = 1, stride=1, padding = 0, bias=False))
        self.layer.add_module('pool', torch.nn.AvgPool1d(kernel_size = 2, stride = 2, padding = 0, ceil_mode=True, count_include_pad=False))
        
    def forward(self, x):
        return self.layer(x)

class DenseNet1d(nn.Module):
    def __init__(self, ch_in, ch_out_final, block_size = [5]*20, growth_rate = 32, bottleneck = 4, compression = 0.5, dropout = 0):
        super(DenseNet1d, self).__init__()
        
        ch_out = 2*growth_rate
        
        self.main = nn.Sequential()
        
        self.main.add_module('conv1', Conv1dLayer(ch_in, ch_out, 7, stride=2, padding = 7/2, dropout = dropout))
        self.main.add_module('pool', torch.nn.AvgPool1d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True, count_include_pad=False))
        
        ch_block_in = ch_out
        
        for i in range(len(block_size)):
            #add a dense block
            self.main.add_module(str(i), DenseBlock1d(ch_block_in, 
                                                      block_size[i], 
                                                      growth_rate = growth_rate, 
                                                      bottleneck = bottleneck, 
                                                      dropout = dropout)) 
                                 
            ch_block_in = ch_block_in + (block_size[i]) * growth_rate
            
            #if not the last layer, add a transfer block
            if i != len(block_size)-1:
                
                ch_block_out = ch_block_in
                
                ch_block_out = int(np.floor(ch_block_out*compression))
                
                self.main.add_module(str(i)+'_transfer', DenseTransfer1d(ch_block_in, ch_block_out))
                
                ch_block_in = ch_block_out
                                 
        self.main.add_module('avg_pool', nn.AdaptiveAvgPool1d(1))
                                     
        self.final = nn.Linear(ch_block_in, ch_out_final)
        
    
    def forward(self, x):
                                 
        x = self.main(x)
        x = torch.squeeze(x)                   
        x = self.final(x)
  
        return x
        
class Seq2Loc(nn.Module):
    def __init__(self, im_ch_in, im_ch_out, string_net):
        super(Seq2Loc, self).__init__()
        
        
        ksize = 4
        dstep = 2
        
        self.string_net = string_net
        
        self.enc = nn.Sequential(
            
            Residual2d(im_ch_in, 64-im_ch_in, ksize, stride = 2, dropout = 0, resid_type = 'cat'),
            nn.BatchNorm2d(64),
        
            nn.ReLU(inplace=True),
            Residual2d(64, 64, ksize, stride = 2, dropout = 0, resid_type = 'cat'),
            nn.BatchNorm2d(128),
        
            nn.ReLU(inplace=True),
            Residual2d(128, 128, ksize, stride = 2, dropout = 0, resid_type = 'cat'),
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            Residual2d(256, 256, ksize, stride = 2, dropout = 0, resid_type = 'cat'),
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            Residual2d(512, 512, ksize, stride = 2, dropout = 0, resid_type = 'cat'),
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            Residual2d(1024, 1024, ksize, stride = 2, dropout = 0, resid_type = 'sum'),
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True)
        )
        
        self.dec = nn.Sequential(
            nn.BatchNorm2d(2048),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2048, 1024, ksize, dstep, 1),
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, ksize, dstep, 1),
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, ksize, dstep, 1),
            nn.BatchNorm2d(128),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, ksize, dstep, 1),
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, im_ch_out, ksize, dstep, 1),

            nn.Sigmoid()             
        )
        
    def forward(self, im_in, seq_in):

        #do the image part
        z_im = self.enc(im_in)
        z_im_size = z_im.shape
        
        #do the string part
        z_seq = self.string_net(seq_in)
        z_seq = torch.unsqueeze(torch.unsqueeze(z_seq,2),2).repeat(1,1,z_im_size[2],z_im_size[3])
        
        #concatenate the string and image parts across the channel dimension
        z = torch.cat((z_im, z_seq), 1)
        
        y_hat = self.dec(z)
        
                
        return y_hat

        
# class SeqConvClassifier(nn.Module):
#     def __init__(self, ch_in, growth_rate):
#         super(SeqConvClassifier, self).__init__()
        
#         kernel_size = 16
#         stride = 2
#         padding = 1
        
#         self.layer = nn.Sequential(
#                                 Conv1dLayer(ch_in, 128, kernel_size, stride, padding),
#                                 Conv1dLayer(128, 256, kernel_size, stride, padding),
#                                 Conv1dLayer(256, 512, kernel_size, stride, padding),
#                                 Conv1dLayer(512, 1024, kernel_size, stride, padding),
#                                 Conv1dLayer(1024, 1024, kernel_size, stride, padding),                        
#                                 )
        
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.out_layer = nn.Linear(1024, ch_out)
        
#     def forward(self, input):
        
#         out = self.layer(input)
#         out = self.pool(out)
#         out = torch.squeeze(out)

#         out = self.out_layer(out)
        
#         return out