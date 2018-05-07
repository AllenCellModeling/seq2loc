from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

import torch
import torch.utils.data 
from torch.autograd import Variable

import pandas as pd
import numpy as np

from seq2loc.utils import *

import math

import pdb

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
aas = ''.join(aas)

nucleotides = 'A','C','T','G'
nucleotides = ''.join(nucleotides)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path = './data/uniprot.csv', max_seq_len = 25, mlb = None, sequence_map = 'aa', column_name = 'GO id short'):
        self.max_seq_len = max_seq_len
        
        df = pd.read_csv(sequence_path)

        
        
        urows = np.unique(df['protID'].astype(str), return_index = True)[1]
        df = df.loc[urows]
        
        df = df.dropna(subset=[column_name])
        
        df = df.reset_index(drop=True)

        
        if mlb is None:
            mlb = MultiLabelBinarizer()
            mlb.fit(df[column_name].str.split(';').tolist())
        
        some_hot_targets = mlb.transform(df[column_name].str.split(';').tolist())
        df['GOsomehot'] = pd.Series(tuple(some_hot_targets.astype(np.float32)))

        df_somehot = df['GOsomehot']
        
        if sequence_map is 'aa':
            self.sequence_map = aas
            df_sequences = df['Sequence']
        elif sequence_map is 'nucleotide':
            self.sequence_map = nucleotides
            df_sequences = df['Sequence_nucleotide']
        
        nan_inds = np.where([isinstance(seq, float) and np.isnan(seq) for seq in df_sequences])[0]
        df_sequences[nan_inds] = ''

        #Trim out the bonkers long sequences
        seq_lengths = np.array([len(seq) for seq in df_sequences])
        max_len = np.percentile(seq_lengths[seq_lengths>0], 99.5)
        
        keep_inds = (seq_lengths <= max_len) & (seq_lengths > 0)
        
        df_sequences = df_sequences[keep_inds].reset_index(drop=True)

        self.somehot = df_somehot[keep_inds].reset_index(drop=True)
        self.pd_sequences = df_sequences
        
        self.mlb = mlb
        
    def __getitem__(self, index):
    
        seq = self.pd_sequences[index]
        
        if len(seq) <= self.max_seq_len:
            tensor_indices = lineToIndices(seq, self.sequence_map)
        else:
            start = np.random.randint(len(seq)-self.max_seq_len)
            
            tensor_indices = lineToIndices(seq[start:(start+self.max_seq_len)], self.sequence_map)
            
        somehot = self.somehot[index]
            
        return tensor_indices, torch.Tensor(somehot)
        
    def __len__(self):
        return len(self.pd_sequences)
    
class PaddedSequenceDataset(torch.utils.data.Dataset):
    #returns tensors in <batch, len, channels> order
        
    def __init__(self, sequenceDataset, GPU_id = None):
        self.sequenceDataset = sequenceDataset
        self.GPU_id = GPU_id
            
    def __getitem__(self, indices):

        sequence_tensor_indices = list()
        somehots = list()

        #get all the sequences as a list of character indices
        for index in indices:
            tensor_indices, somehot = self.sequenceDataset[index]
            sequence_tensor_indices += [Variable(tensor_indices)]
            somehots += [Variable(somehot)]

        #get the longest sequence
        ind = np.argmax([len(s) for s in sequence_tensor_indices])

        tensor_len = len(sequence_tensor_indices[ind])
        nchars = len(self.sequenceDataset.sequence_map)
        
        #pad all shorter sequences with the empty character
        sequence_tensors = list()
        
        for i in range(len(sequence_tensor_indices)):

            my_inds = sequence_tensor_indices[i]
            my_len = my_inds.shape[0]
            additional_len = tensor_len - my_len
            

            my_inds = torch.unsqueeze(my_inds, 1)
            
            my_inds[my_inds == -1] = 0
            
            try:
                sequence_tensors += [torch.cat([indicesToTensor(my_inds, ndims = nchars), torch.zeros([additional_len, 1, nchars])], 0)]
            except:
                pdb.set_trace()

        sequence_tensors = Variable(torch.cat(sequence_tensors, 1))
        somehots = torch.stack(somehots)

        sequence_tensors = sequence_tensors.transpose(1,0).transpose(2,1)

        if self.GPU_id is not None:
            sequence_tensors = sequence_tensors.cuda(self.GPU_id)
            somehots = somehots.cuda(self.GPU_id)
        return sequence_tensors, somehots            

    def __len__(self):
        return len(self.sequenceDataset)

    