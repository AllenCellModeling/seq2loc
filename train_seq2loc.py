import argparse
import os
import json
import torch
import numpy as np

from tensorboardX import SummaryWriter

import seq2loc
import seq2loc.models
import seq2loc.train_seq2loc
from seq2loc.datasets import Seq2LocDataset

import seq2loc.models.tiramisu.tiramisu_seq2loc as tiramisu_seq2loc

import pdb

import os

import torch.optim as optim


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--GPU_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--my_seed', type=int, default=0, help='random seed')

parser.add_argument('--save_progress_iter', type=int, default=1, help='number of epochs between saving progress')
parser.add_argument('--save_state_iter', type=int, default=10, help='number of epochs between saving state')

parser.add_argument('--max_seq_len', type=int, default=20000, help='randomly trim sequences to this length')
parser.add_argument('--patch_size', type=int, default=-1, help='patch size for the dataset')

parser.add_argument('--model', type=str, default='seq2loc', help='name of model to use')
parser.add_argument('--model_seq', type=str, default='seq2loc', help='name of sequence model to use')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nepochs', type=int, default=5000, help='total number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--loss', type=str, default='L1Loss', help='loss type')
parser.add_argument('--amsgrad', type=str2bool, default=False, help='use AMSGrad variant of ADAM')

parser.add_argument('--im_growth_rate', type=int, default=16, help='growth rate of the tiramisu net')



parser.add_argument('--seq_layers_deep', type=int, default=20, help ='number of layers in the model')
parser.add_argument('--seq_ch_intermed', type=int, default=256, help='number of intermediate channels between layers')
parser.add_argument('--seq_resid', type=str, default='sum', help = 'residual type of network, "sum" or "cat"')
parser.add_argument('--seq_dropout', type=float, default=0.5, help='dropout rate for sequence portion of network')
parser.add_argument('--seq_nout', type=int, default=1024, help='output size of the sequence model')
parser.add_argument('--seq_pooling', type=str, default='max', help='pooling type to use in the network, "max" or "avg"')

opts = parser.parse_args()
print(opts)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ID) for ID in opts.GPU_ids])
GPU_ids = list(range(0, len(opts.GPU_ids)))
GPU_id = GPU_ids[0]

torch.manual_seed(opts.my_seed)
torch.cuda.manual_seed(opts.my_seed)
np.random.seed(opts.my_seed)

writer = SummaryWriter()
save_dir = writer.file_writer.get_logdir()

#Save the preferences
with open('{}/prefs.json'.format(save_dir), 'w') as fp:
    
    # pdb.set_trace()
    json.dump(vars(opts), fp)



    
ds = Seq2LocDataset('./data/hpa_data_resized_train.csv', 
                    max_seq_len = opts.max_seq_len, 
                    patch_size = opts.patch_size,
                    GPU_id = GPU_id)

ds_validate = Seq2LocDataset('./data/hpa_data_resized_validate.csv', 
                    max_seq_len = opts.max_seq_len, 
                    patch_size = opts.patch_size,
                    GPU_id = GPU_id)


N_LETTERS = len(ds.sequence_map)
N_CLASSES = opts.seq_nout

criterion = getattr(torch.nn, opts.loss)()


if opts.model_seq == 'transformer':
    model_seq = seq2loc.models.TransformerClassifier(N_LETTERS, N_CLASSES, 
                                                     growth_rate = 64, 
                                                     max_seq_len = 20000,
                                                     n_layers = opts.seq_layers_deep,
                                                     n_heads_per_layer = 6).cuda(GPU_id)
else:
    model_seq = seq2loc.models.SeqConvResidClassifier(N_LETTERS, N_CLASSES, 
                                                  kernel_size = 3, 
                                                  layers_deep = opts.seq_layers_deep, 
                                                  ch_intermed = opts.seq_ch_intermed, 
                                                  pooling_type = opts.seq_pooling,
                                                  resid_type = opts.seq_resid,
                                                  dropout = opts.seq_dropout,
                                                  downsize_on_nth = 3).cuda(GPU_id)


if opts.model == 'tiramisu':
    model = tiramisu_seq2loc.FCDenseNet(in_channels = 2, 
                                        down_blocks=(3,3,3,3,3),
                                        up_blocks=(3,3,3,3,3),
                                        bottleneck_layers=5,
                                        growth_rate = opts.im_growth_rate,
                                        string_net=model_seq)
elif opts.model == 'tiramisu_simple':
    model = tiramisu_seq2loc.FCDenseNet_simple(in_channels = 2, 
                                        down_blocks=(3,3,3,3,3),
                                        up_blocks=(3,3,3,3,3),
                                        bottleneck_layers=5,
                                        growth_rate = opts.im_growth_rate,
                                        string_net=model_seq)
    
else:
    model = seq2loc.models.Seq2Loc(2, 1, model_seq)
    
model = model.cuda(GPU_id)

    
    
model.apply(seq2loc.utils.model.weights_init)

opt = optim.Adam(model.parameters(), lr = opts.lr, betas=(0.5, 0.999), amsgrad=opts.amsgrad)


model = seq2loc.train_seq2loc.train(model, opt, criterion, ds, ds_validate, writer = writer, nepochs = opts.nepochs, batch_size = opts.batch_size)


