import argparse
import os
import json
import torch
import numpy as np

from tensorboardX import SummaryWriter

import seq2loc
import seq2loc.models
from seq2loc.datasets import SequenceDataset, PaddedSequenceDataset, NewsgroupsDataset

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
parser.add_argument('--sequence_type', type=str, default='aa', help='sequence type to train on, can be "aa", or "nucleotide"')
parser.add_argument('--trim_to_firstlast', type=str2bool, default=False, help='use the first and last 100 elements of the sequence (instead of the whole sequence)')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--kernel_size', type=int, default=8, help='size of convolution kernel')
parser.add_argument('--layers_deep', type=int, default=8, help='how many layers deep')
parser.add_argument('--ch_intermed', type=int, default=128, help='output layers for each block in the residual network')
parser.add_argument('--pooling_type', type=str, default='max', help='type of pooling in the last layer of the network')
parser.add_argument('--resid_type', type=str, default='cat', help='how to join residual outputs, ("cat", or "sum")')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--downsize_on_nth', type=int, default=1, help='downsize layer after this many iterations')

parser.add_argument('--model_type', type=str, default=None, help='alternative model types')

parser.add_argument('--column_name', type=str, default='GO id short', help='column name from the dataframe')


parser.add_argument('--nepochs', type=int, default=5000, help='total number of epochs')

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


if opts.sequence_type == 'newsgroups':
    ds = PaddedSequenceDataset(NewsgroupsDataset('./data/20_newsgroups/train.pickle'), GPU_id = GPU_id)
    ds_validate = PaddedSequenceDataset(NewsgroupsDataset('./data/20_newsgroups/validate.pickle', 
                                                    mlb = ds.sequenceDataset.mlb), GPU_id = GPU_id)
else:
    
    ds = PaddedSequenceDataset(SequenceDataset('./data/hpa_data_resized_train.csv', 
                                               max_seq_len = opts.max_seq_len, 
                                               sequence_map = opts.sequence_type, 
                                               column_name = opts.column_name,
                                               trim_to_firstlast = opts.trim_to_firstlast), GPU_id = GPU_id)

    ds_validate = PaddedSequenceDataset(SequenceDataset('./data/hpa_data_resized_validate.csv', 
                                                    max_seq_len = opts.max_seq_len, 
                                                    mlb = ds.sequenceDataset.mlb, 
                                                    sequence_map = opts.sequence_type, 
                                                    column_name = opts.column_name,
                                                    trim_to_firstlast = opts.trim_to_firstlast), GPU_id = GPU_id)

N_LETTERS = len(ds.sequenceDataset.sequence_map)
N_CLASSES = len(ds.sequenceDataset.mlb.classes_)

criterion = torch.nn.BCEWithLogitsLoss()

if opts.model_type == 'densenet':
    model = seq2loc.models.DenseNet1d(N_LETTERS, N_CLASSES, dropout = opts.dropout).cuda(GPU_id)
else:
    model = seq2loc.models.SeqConvResidClassifier(N_LETTERS, N_CLASSES, 
                                                  kernel_size = opts.kernel_size, 
                                                  layers_deep = opts.layers_deep, 
                                                  ch_intermed = opts.ch_intermed, 
                                                  pooling_type = opts.pooling_type,
                                                  resid_type = opts.resid_type,
                                                  dropout = opts.dropout,
                                                 downsize_on_nth = opts.downsize_on_nth).cuda(GPU_id)


model.apply(seq2loc.utils.model.weights_init)

opt = optim.Adam(model.parameters(), lr = opts.lr)



model = seq2loc.train(model, opt, criterion, ds, ds_validate, writer = writer, nepochs = opts.nepochs, batch_size = opts.batch_size)


