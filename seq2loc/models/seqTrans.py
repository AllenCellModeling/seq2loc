import torch
import transformer
from transformer.Layers import EncoderLayer
from transformer.Models import get_attn_padding_mask

from torch.autograd import Variable
import torch.nn as nn

import pdb


class TransformerLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 5, stride = 5, padding = 2, dropout = 0, 
                 n_head = 8,  d_k = 128, d_v = 128, max_seq_len = 2000):
        super(TransformerLayer, self).__init__()
        
        self.conv_layer = nn.Sequential()
        if dropout>0:
            self.conv_layer.add_module('dr', nn.Dropout(dropout))
        self.conv_layer.add_module('conv', nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size, stride = stride, padding = padding, bias=False))
        self.conv_layer.add_module('bn', nn.BatchNorm1d(ch_out))
        self.conv_layer.add_module('relu', nn.ReLU(inplace=True))
        
        self.transformer = transformer.Models.EncoderSimple(n_max_seq=2000, n_layers=1, n_head=n_head, d_k=d_k, d_v=d_v,
            d_word_vec=ch_out, d_model=ch_out, d_inner_hid=512, dropout=0.1)
        
        self.final = nn.Sequential()
        self.final.add_module('bn2', nn.BatchNorm1d(ch_out))
        self.final.add_module('relu2', nn.ReLU(inplace=True))
        
    def forward(self, x):

        x = self.conv_layer(x)
        x = x.permute(0,2,1).clone()
        x, attn = self.transformer(x, return_attns=True)
        x = x.permute(0,2,1).contiguous().clone()
        out = self.final(x)
        
        return out
        
        
class TransformerClassifier(nn.Module):
    def __init__(self, ch_in, ch_out, growth_rate, max_seq_len, n_layers = 4, n_heads_per_layer = 4, dropout = 0.1, pooling_type = 'avg'):
        super(TransformerClassifier, self).__init__()
 
        ch_out_tmp = growth_rate
        
        self.model = nn.Sequential()
        for i in range(n_layers):

            self.model.add_module('layer'+str(i), TransformerLayer(ch_in, ch_out_tmp, n_head = n_heads_per_layer, dropout=dropout))

            ch_in = growth_rate * (i+1)
            ch_out_tmp = growth_rate * (i+2)
            
        if pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError('pooling_type must be "max" or "avg"')
            
        self.out_layer = nn.Linear(ch_in, ch_out)

    def forward(self, x):
    
        x = self.model(x)
        x = self.pool(x)
        x = torch.squeeze(x)

        out = self.out_layer(x)
        
        return out
    
    


    