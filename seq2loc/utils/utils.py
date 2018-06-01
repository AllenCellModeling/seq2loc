import torch
import numpy as np

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
aas = ''.join(aas)

N_LETTERS = len(aas)

def chars_to_dict(stringmap):
    mydict = {}
    for i, char in enumerate(list(stringmap)):
        mydict[char] = i
        
    return mydict

def letterToIndex(letter, stringmap = aas):
    return stringmap[letter]

def lineToIndices(line, stringmap = aas):
    indices = torch.zeros(len(line)).long()
    for li, letter in enumerate(line):
        index = letterToIndex(letter, stringmap)
        indices[li] = index
        
    return indices

def tensorToChar(tensor, stringmap = aas):
    m, indices = torch.max(tensor, 2)
    
    chars = np.array(list(aas))[indices.cpu().numpy()]
    
    return chars

def indicesToTensor(indices, ndims = N_LETTERS):
    #aka indices to onehot
    
    tensor = torch.zeros(indices.shape[0], indices.shape[1], ndims)
    tensor.scatter_(2, torch.unsqueeze(indices, 2).data, 1)
    
    return tensor

def letterToTensor(letter, stringmap = aas):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letterToIndex(letter, stringmap)] = 1
    return tensor

def lineToTensor(line, ndims = N_LETTERS):
    #aka string to onehot
    tensor = torch.zeros(len(line), 1, ndims)
    tensor_indices = lineToIndices(line)
    
    for li, index in enumerate(tensor_indices):
        tensor[li][0][index] = 1
        
    return tensor

def stopChar(batch_size = 1, ndims = N_LETTERS):
    tensor = torch.zeros(1, batch_size, ndims )
    tensor[0,:,-1] = 1
    
    return tensor

def stopCharIndex():
    return N_LETTERS-1

def n_letters():
    return N_LETTERS

def get_epoch_inds(n_samples, batch_size):

    sample_order = list(range(n_samples))
    np.random.shuffle(sample_order)

    spacing = np.arange(0, n_samples, batch_size)

    inds = [sample_order[spacing[i]:spacing[i+1]] for i in range(len(spacing)-1)]
    
    return inds

def split_dat(uniprot_tsv_path, train_test_split = [0.9, 0.1]):
    #split on unique protein id
    
    pd = pd.read_csv(uniprot_tsv_path, sep = '\t')
    
    pdb.set_trace()
    
def seq_to_padded_tensor(sequence_indices, tensor_len, nchars):
    my_inds = sequence_indices
    my_len = my_inds.shape[0]
    additional_len = tensor_len - my_len


    my_inds = torch.unsqueeze(my_inds, 1)

    my_inds[my_inds == -1] = 0

    sequence_tensor = torch.cat([indicesToTensor(my_inds, ndims = nchars), torch.zeros([additional_len, 1, nchars])], 0)

    return sequence_tensor  
        