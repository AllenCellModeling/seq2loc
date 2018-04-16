import torch
import torch.utils.data 
from torch.autograd import Variable

import pandas as pd
import numpy as np

from seq2loc.utils import *




class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path = './data/uniprot.tsv', max_seq_len = 25):
        self.max_seq_len = max_seq_len
        
        df_uniprot = pd.read_csv(sequence_path, sep='\t')

        pd_sequences = df_uniprot['Sequence']

        #Trim out the bonkers long sequences
        seq_lengths = [len(seq) for seq in pd_sequences]
        max_len = np.percentile(seq_lengths, 99.5)
        pd_sequences = pd_sequences[seq_lengths <= max_len].reset_index(drop=True)

        self.pd_sequences = pd_sequences
        
    def __getitem__(self, index):
    
        seq = self.pd_sequences[index]
        
        if len(seq) <= self.max_seq_len:
            tensor_indices = lineToIndices(seq)
        else:
            start = np.random.randint(len(seq)-self.max_seq_len)
            
            tensor_indices = lineToIndices(seq[start:(start+self.max_seq_len)])
            
        return tensor_indices
        
    def __len__(self):
        return len(self.pd_sequences)
    
class PaddedSequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequenceDataset, GPU_id = None):
            self.sequenceDataset = sequenceDataset
            self.GPU_id = GPU_id
            
        def __getitem__(self, indices):
            
            sequence_tensor_indices = list()
            
            #get all the sequences as a list of character indices
            for index in indices:
                tensor_indices = self.sequenceDataset[index]
                sequence_tensor_indices += [Variable(tensor_indices)]
                
            #sort by length
            inds = np.argsort([len(s) for s in sequence_tensor_indices])[-1::-1]
            sequence_tensor_indices = [sequence_tensor_indices[i] for i in inds]

            #get the longest sequence
            tensor_len = len(sequence_tensor_indices[0])
            nchars = n_letters()
            
            #pad all shorter sequences with the stop character
            for i in range(len(sequence_tensor_indices)):
                
                my_inds = sequence_tensor_indices[i]
                my_len = my_inds.shape[0]
                sequence_tensor_indices[i] = torch.unsqueeze(torch.cat([my_inds, Variable(torch.ones(tensor_len - my_len).long()*(nchars-1))]), 1)
            
            sequence_tensor_indices = torch.cat(sequence_tensor_indices, 1)
            sequence_tensors = Variable(indicesToTensor(sequence_tensor_indices))
            
            if self.GPU_id is not None:
                sequence_tensors = sequence_tensors.cuda(self.GPU_id)
                sequence_tensor_indices = sequence_tensor_indices.cuda(self.GPU_id)
                
            return sequence_tensors, sequence_tensor_indices            
            
        def __len__(self):
            return len(self.sequenceDataset)

    