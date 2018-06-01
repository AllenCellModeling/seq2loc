import torch
import torch.nn as nn

from .layers import *

import pdb

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, string_net = None):
        super().__init__()

        #####################
        # Get the outputs of string_net
        
        self.string_net = string_net
    
        try:
            string_net_out_size =self.string_net.out_layer.out_features
        except:
            string_net_out_size = 0
        #####################
        
        
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################


        
        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        
        
        # prev_block_channels += string_net_out_size
        # cur_channels_count += string_net_out_size
        
        self.sequenceHeads = nn.ModuleList([])
        
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            
            nSeqHeadOut =  int(string_net_out_size/(2**i))
            self.sequenceHeads.append(LinearLayer(string_net_out_size,nSeqHeadOut))
            
            self.transUpBlocks.append(TransitionUp(prev_block_channels + nSeqHeadOut, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                                    cur_channels_count, growth_rate, up_blocks[i],
                                        upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        nSeqHeadOut =  int(string_net_out_size/(2**(i+1)))
        self.sequenceHeads.append(LinearLayer(string_net_out_size,nSeqHeadOut))
        
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels + nSeqHeadOut, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=1, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, im_in, seq_in):
    
        out = self.firstconv(im_in)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
      
        # pdb.set_trace()
        
        #do the string part
        z_seq_tmp = self.string_net(seq_in)
        
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            
            #concatenate the output from the sequence net to all upward moving blocks
            z_im_size = out.shape
                                      
            z_seq = self.sequenceHeads[i](z_seq_tmp)
            z_seq = torch.unsqueeze(torch.unsqueeze(z_seq ,2),2)        
            z_seq = z_seq.repeat(1,1,z_im_size[2],z_im_size[3])
            out = torch.cat((out, z_seq), 1)
            
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.sigmoid(out)
        return out
    
    

class FCDenseNet_simple(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, string_net = None):
        super().__init__()

        #####################
        # Get the outputs of string_net
        
        self.string_net = string_net
    
        try:
            string_net_out_size =self.string_net.out_layer.out_features
        except:
            string_net_out_size = 0
        #####################
        
        
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        growth_rate_bottleneck = growth_rate * 4
        self.add_module('bottleneck',Bottleneck(cur_channels_count+string_net_out_size,
                                     growth_rate_bottleneck, bottleneck_layers))
        
        prev_block_channels = growth_rate_bottleneck*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        
        
        # prev_block_channels += string_net_out_size
        # cur_channels_count += string_net_out_size
        
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                                    cur_channels_count, growth_rate, up_blocks[i],
                                        upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=1, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, im_in, seq_in):
    
        out = self.firstconv(im_in)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        #concatenate the sequence output into the last bottleneck
        z_im_size = out.size()
        z_seq_tmp = self.string_net(seq_in)
        
        z_seq = torch.unsqueeze(torch.unsqueeze(z_seq_tmp ,2),2)        
        z_seq = z_seq.repeat(1,1,z_im_size[2],z_im_size[3])
        out = torch.cat((out, z_seq), 1)
        
        out = self.bottleneck(out)
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.sigmoid(out)
        return out

