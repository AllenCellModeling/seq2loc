import numpy as np
import seq2loc.utils.model as model_utils
import seq2loc.utils as utils
import torch
import torch.nn as nn

import torchvision.utils as vutils

from sklearn.metrics import average_precision_score, roc_auc_score

from tqdm import tqdm as tqdm
from decimal import Decimal

import pdb

def train(model, opt, criterion, ds, ds_validate, writer, nepochs, batch_size, save_progress_epoch = 1, save_state_epoch = 10):
    
    save_dir = writer.file_writer.get_logdir()

    iteration = 0

    for epoch in range(nepochs):

        epoch_inds = utils.get_epoch_inds(len(ds), batch_size)
        pbar = tqdm(epoch_inds)

        epoch_losses = list()

        y_list = list()
        y_hat_list = list()

        for batch in pbar:
            opt.zero_grad()

            data = ds[batch] 

            x_im = data['image_source']
            x_str = data['sequence_source']
            y = data['image_target']

            y_hat  = model(x_im, x_str)

            loss = criterion(y_hat, y)

            loss.backward()
            opt.step()

            losses_np = np.squeeze(loss.detach().cpu().numpy())

            epoch_losses += [losses_np]
            pbar.set_description('%.4E' % Decimal(str(losses_np)))

            writer.add_scalar('loss/train', losses_np, iteration)

            iteration += 1
            
            
        ###########################
        ### Write out test results
        ###########################

        write_progress(model, ds, writer, iteration, 'train')

        if epoch % save_progress_epoch == 0:
            save_progress(model, criterion, batch_size, ds_validate, writer, iteration)

        if epoch % save_state_epoch == 0:
            model_utils.save_state(model, opt, '{}/model.pyt'.format(save_dir))
            
        pbar.set_description('%.4E' % Decimal(str(np.mean(epoch_losses))))

def save_progress(model, criterion, batch_size, ds, writer, iteration):
    model.train(False)
    # ds.train(False)


    epoch_inds = utils.get_epoch_inds(len(ds), batch_size)

    y_list = list()
    y_hat_list = list()
    losses_test = list()

    for batch in epoch_inds:

        data = ds[batch] 

        x_im = data['image_source']
        x_str = data['sequence_source']
        y = data['image_target']

        with torch.no_grad():
            y_hat  = model(x_im, x_str)

        loss = criterion(y_hat, y)
        losses_test += [np.squeeze(loss.detach().cpu().numpy())]


        y_list += [y.data.cpu().numpy()]
        y_hat_list += [y_hat.data.cpu().numpy()]
    
    writer.add_scalar('loss/test', np.mean(losses_test), iteration)

    write_progress(model, ds, writer, iteration, train_or_test = 'test')

    model.train(True) 
    # ds.train(False)

    
def write_progress(model, ds, writer, iteration, train_or_test = 'train'):

    batch_size = 8
    
    model.train(False)  
    ds.train(False)
    
    batch = list(range(batch_size))
    
    data = ds[batch]
                 
    x_im = data['image_source']
    x_str = data['sequence_source']
    y = data['image_target']

    with torch.no_grad():
        y_hat  = model(x_im, x_str)
                 
    y_hat = y_hat.data.cpu()
    y_hat = vutils.make_grid(y_hat, normalize=True, scale_each=False)
                 
    writer.add_image('image/{}'.format(train_or_test), y_hat, iteration)
                 
    model.train(True) 
    ds.train(True)