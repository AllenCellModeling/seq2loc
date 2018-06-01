from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer

import torch
import torch.utils.data 
from torch.autograd import Variable

import pandas as pd
import numpy as np

from seq2loc.utils import *

import math

import pickle
import tqdm

import os

from skimage import io



import pdb

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
aas = ''.join(aas)

nucleotides = 'A','C','T','G'
nucleotides = ''.join(nucleotides)


from joblib import Parallel, delayed

class NewsgroupsDataset(torch.utils.data.Dataset):
    def __init__(self, pickle_path = './data/20_newsgroups/train.pickle', max_seq_len = 20000, mlb = None):
        
        column_name = 'group'
        target_name = 'text'
        
        self.sequence_map = chars_to_dict(''.join(['\x02' '\x03' '\x06' '\x08' '\t' '\n' '\x0c' '\x18' '\x19' '\x1a' '\x1b',
                                     '\x1c' '\x1e' ' ' '!' '"' '#' '$' '%' '&' "'" '(' ')' '*' '+' ',' '-' '.',
                                     '/' '0' '1' '2' '3' '4' '5' '6' '7' '8' '9' ':' ';' '<' '=' '>' '?' '@',
                                     'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R',
                                     'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' '[' '\\' ']' '^' '_' '`' 'a' 'b' 'c' 'd',
                                     'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v',
                                     'w' 'x' 'y' 'z' '{' '|' '}' '~' '\x7f']))
        
        
        
        self.max_seq_len = max_seq_len
        
        with open(pickle_path, 'rb') as handle:
            df = pickle.load(handle)
    
        if mlb is None:
            mlb = LabelBinarizer()
            mlb.fit(df[column_name])
        
        self.max_seq_len = max_seq_len
        self.somehot = mlb.transform(df[column_name])
        
        self.sequences = [lineToIndices(seq, self.sequence_map) for seq in tqdm.tqdm(df['text'].tolist())]
        
        # self.sequences = df['text'].tolist()
        self.mlb = mlb
    
    def __getitem__(self, index):
    
        seq = self.sequences[index]
        
        if len(seq) > self.max_seq_len:
            start = np.random.randint(len(seq)-self.max_seq_len)
            seq = seq[start:(start+self.max_seq_len)]
        
        
        # tensor_indices = lineToIndices(seq, self.sequence_map)
            
        somehot = self.somehot[index]
            
        return seq, torch.Tensor(somehot)
        
    def __len__(self):
        return len(self.sequences)    

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path = './data/uniprot.csv', max_seq_len = 25, mlb = None, sequence_map = 'aa', column_name = 'GO id short', trim_to_firstlast = False):
        self.max_seq_len = max_seq_len
        
        if isinstance(sequence_path, str):
            df = pd.read_csv(sequence_path)
        else: #assume it is a dataframe
            df = sequence_path
        

        
        
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
            sequence_map = chars_to_dict(aas)
            df_sequences = df['Sequence']
        elif sequence_map is 'nucleotide':
            sequence_map = chars_to_dict(nucleotides)
            df_sequences = df['Sequence_nucleotide']
        
        self.sequence_map = sequence_map 
        
        nan_inds = np.where([isinstance(seq, float) and np.isnan(seq) for seq in df_sequences])[0]
        df_sequences[nan_inds] = ''

        #Trim out the bonkers long sequences
        seq_lengths = np.array([len(seq) for seq in df_sequences])
        max_len = np.percentile(seq_lengths[seq_lengths>0], 99.5)
        
        keep_inds = (seq_lengths <= max_len) & (seq_lengths > 0)
        
        df_sequences = df_sequences[keep_inds].reset_index(drop=True)

        self.somehot = df_somehot[keep_inds].reset_index(drop=True)
        self.pd_sequences = df_sequences
        
        self.trim_to_firstlast = trim_to_firstlast
        
        self.mlb = mlb
        
    def __getitem__(self, index):
    
        seq = self.pd_sequences[index]
        
        if self.trim_to_firstlast:
            seq = ''.join([seq[0:100], seq[-100:]])
        
        if len(seq) > self.max_seq_len:
            start = np.random.randint(len(seq)-self.max_seq_len)
            seq = seq[start:(start+self.max_seq_len)]
            

            
        tensor_indices = lineToIndices(seq, self.sequence_map)
            
        somehot = self.somehot[index]
            
        return tensor_indices, torch.Tensor(somehot)
        
    def __len__(self):
        return len(self.pd_sequences)
    
class PaddedSequenceDataset(torch.utils.data.Dataset):
    #returns tensors in <batch, len, channels> order
        
    def __init__(self, sequenceDataset, GPU_id = None, n_jobs = 8):
        self.sequenceDataset = sequenceDataset
        self.GPU_id = GPU_id
            
        self.parallel = Parallel(n_jobs == n_jobs)
            

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
        
        sequence_tensors = self.parallel(delayed(seq_to_padded_tensor)(sequence_inds, tensor_len, nchars) for sequence_inds in sequence_tensor_indices)
        
        sequence_tensors = Variable(torch.cat(sequence_tensors, 1))
        somehots = torch.stack(somehots)

        sequence_tensors = sequence_tensors.transpose(1,0).transpose(2,1)

        if self.GPU_id is not None:
            sequence_tensors = sequence_tensors.cuda(self.GPU_id)
            somehots = somehots.cuda(self.GPU_id)
        return sequence_tensors, somehots            

    def __len__(self):
        return len(self.sequenceDataset)

    
class Seq2LocDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path = './data/hpa_data_resized_train.csv', max_seq_len = 25, trim_to_firstlast = False, n_jobs = 8, GPU_id = None, patch_size = -1):
        
        
        self.df = pd.read_csv(sequence_path)
        self.GPU_id = GPU_id
        
        self.sequence_map = chars_to_dict(aas)
        
        self.parallel = Parallel(n_jobs == n_jobs)
        self.patch_size = patch_size
        
        self.is_train = True
    
    def __getitem__(self, indices):
        
        sequence_tensor_indices = list()
        images_source = list()
        images_target = list()

        #get all the sequences as a list of character indices
        for index in indices:
            df_row = self.df.iloc[index]
            tensor_indices = lineToIndices(df_row['Sequence'], self.sequence_map)
            sequence_tensor_indices += [Variable(tensor_indices)]

            
            im_path = os.sep.join(['./data/hpa/', df_row['ENSG'], df_row['RGB256px']])
            image = io.imread(im_path).astype(float)/255
            
            # pdb.set_trace()
            # if np.any(np.max(image,2) == 0):
            #     continue
                
            image = image.transpose([2,0,1])
                        
            for i in range(len(image)):
                if np.max(image[i]) > 0:
                    image[i] = image[i]/np.max(image[i])
                
            # images += [image]
            
            if self.patch_size > 0 and self.is_train:
                patch_size = self.patch_size
                imsize = torch.Tensor(image.shape[1:])
                
                start = [np.random.randint(dim- patch_size) for dim in imsize]
                
                image = image[:, start[0]:(start[0]+patch_size), start[1]:(start[1]+patch_size)]
                
            
            images_source += [np.expand_dims(image[[0,2]], 0)]
            images_target += [np.expand_dims(np.expand_dims(image[1],0), 0)]

            
        #get the longest sequence
        ind = np.argmax([len(s) for s in sequence_tensor_indices])

        tensor_len = len(sequence_tensor_indices[ind])
        nchars = len(self.sequence_map)
        
        #pad all shorter sequences with the empty character
        sequence_tensors = list()
        
        sequence_tensors = self.parallel(delayed(seq_to_padded_tensor)(sequence_inds, tensor_len, nchars) for sequence_inds in sequence_tensor_indices)
        sequence_tensors = Variable(torch.cat(sequence_tensors, 1))
        sequence_tensors = sequence_tensors.transpose(1,0).transpose(2,1)
        
        
        images_source = Variable(torch.cat([torch.Tensor(image).float() for image in images_source], 0))
        images_target = Variable(torch.cat([torch.Tensor(image).float() for image in images_target], 0))
        

        if self.GPU_id is not None:
            images_source = images_source.cuda(self.GPU_id)
            images_target = images_target.cuda(self.GPU_id)
            sequence_tensors = sequence_tensors.cuda(self.GPU_id)
            
        return {'image_source': images_source, 'sequence_source': sequence_tensors, 'image_target': images_target}
    
    def train(self, true_or_false):
        self.is_train = true_or_false
    
    def __len__(self):
        return len(self.df)