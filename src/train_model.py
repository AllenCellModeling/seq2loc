import torch
from torch import nn
from torch.autograd import Variable

from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
from tqdm import tqdm, tqdm_notebook
from model_analysis import torch_confusion_matrix

def train_model(net,
                dataprovider,
                class_weights=None,
                class_names=None,
                N_epochs=1000,
                phases=('train', 'test'),
                learning_rate=1e-4,
                gpu_id=0,
                print_freq=100):
    
    # tensorboard logger
    writer = SummaryWriter()
        
    # put model on gpu if desired
    if gpu_id is not None:
        net = net.cuda(gpu_id)
    
    # optimization criterion and optimizer -- should pass in as args
    criterion = nn.CrossEntropyLoss() if class_weights is None else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

    # makes progress bar nice
    batch_size = dataprovider.opts['dataloader_kwargs']['train']['batch_size']
    epoch_len = sum([len(dataprovider.splits[split]) - len(dataprovider.splits[split]) % batch_size for split in dataprovider.splits.keys()])
    
    # dict of confusion matrices
    cmat = {}
    
    # pretty progress bar
    with tqdm_notebook(total=N_epochs*epoch_len, unit=' images') as pbar:
        
        # cycle through data multiple times
        for epoch in range(N_epochs):
            
            # pretty progress bar
            pbar.set_description('Epoch {}'.format(epoch+1))
            
            # track predictions and loss for logging
            mito_labels = {k:{'true_labels':[], 'pred_labels':[]} for k in dataprovider.dataloaders.keys()}            
            ave_loss = {phase:0.0 for phase in phases}
            
            # train, test, etc
            for phase in phases:
                
                # make sure network is in the correct mode
                net.train() if phase == 'train' else net.eval()

                for i, minibatch in enumerate(dataprovider.dataloaders[phase]):
                    
                    # make samples variables and put on gpu if desired
                    inputs, labels = Variable(minibatch['inputImage']), Variable(minibatch['cellLine'])
                    if gpu_id is not None:
                        inputs, labels = inputs.cuda(gpu_id), labels.cuda(gpu_id)

                    # pass data through model and compute loss
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    
                    # backprop and step gradients if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # track predictions for logging
                    _, pred = outputs.max(1)
                    mito_labels[phase]['true_labels'] += list(labels.data.cpu().squeeze().numpy())
                    mito_labels[phase]['pred_labels'] += list(pred.data.cpu().numpy())
                    
                    # track average loss for progress bar
                    ave_loss[phase] = (i*ave_loss[phase] + loss.data.item())/(i+1)
                    pbar.update(batch_size)
                    pbar.set_postfix(phase=phase, loss=ave_loss[phase])
                    
                # write out loss every epoch
                writer.add_scalar('loss/{}'.format(phase),  loss.data.item(), epoch)
                
                # write out confusion matrix every epoch
                cmat[phase] = torch_confusion_matrix(mito_labels[phase]['true_labels'], mito_labels[phase]['pred_labels'])
                writer.add_image('confusion_matrix/{}'.format(phase), cmat[phase], epoch)
                
                # write out prec/recall every epoch
                precision, recall, _, _ = precision_recall_fscore_support(mito_labels[phase]['true_labels'], mito_labels[phase]['pred_labels'], warn_for=())
                writer.add_scalars('precision/{}'.format(phase), dict(zip(class_names,precision)), epoch)
                writer.add_scalars('recall/{}'.format(phase), dict(zip(class_names,recall)), epoch)
                
            # write out weights for last later to make a histogram
            for name, param in net.fc.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    
    return net