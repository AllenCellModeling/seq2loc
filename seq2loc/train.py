import numpy as np
import seq2loc.utils.model as model_utils
import seq2loc.utils as utils
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score

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

            x, y = ds[batch]        

            y_hat  = model(x)

            loss = criterion(y_hat, y)

            loss.backward()
            opt.step()

            losses_np = np.squeeze(loss.detach().cpu().numpy())

            epoch_losses += [losses_np]
            pbar.set_description('%.4E' % Decimal(str(losses_np)))

            writer.add_scalar('loss/train', losses_np, iteration)

            iteration += 1
            
            y_list += [y.data.cpu().numpy()]
            y_hat_list += [y_hat.data.cpu().numpy()]

            
        ###########################
        ### Write out test results
        ###########################
        y_list = np.vstack(y_list)
        y_hat_list = np.vstack(y_hat_list)

        write_progress(writer, iteration, y_list, y_hat_list, 'train')
        
        if epoch % save_progress_epoch == 0:
            save_progress(model, criterion, batch_size, ds_validate, writer, iteration)
            
        if epoch % save_state_epoch == 0:
            model_utils.save_state(model, opt, '{}/model.pyt'.format(save_dir))
            
        pbar.set_description('%.4E' % Decimal(str(np.mean(epoch_losses))))

def save_progress(model, criterion, batch_size, ds_validate, writer, iteration):
    model.train(False)

    epoch_inds = utils.get_epoch_inds(len(ds_validate), batch_size)

    y_list = list()
    y_hat_list = list()
    losses_test = list()

    for batch in epoch_inds:

        x, y = ds_validate[batch]        

        with torch.no_grad():
            y_hat  = nn.Sigmoid()(model(x))

        loss = criterion(y_hat, y)
        losses_test += [np.squeeze(loss.detach().cpu().numpy())]


        y_list += [y.data.cpu().numpy()]
        y_hat_list += [y_hat.data.cpu().numpy()]
    
    writer.add_scalar('loss/test', np.mean(losses_test), iteration)

    # track predictions for logging
    y_list = np.vstack(y_list)
    y_hat_list = np.vstack(y_hat_list)

    write_progress(writer, iteration, y_list, y_hat_list, train_or_test = 'test')

    model.train(True) 

    
def write_progress(writer, iteration, true_labs, pred_acts, train_or_test = 'train'):
    # compute and write out area under the precision recall curve every minibatch
    aucpr_dict = {str(col):average_precision_score(true_labs[:,i], pred_acts[:,i]) for i,col in enumerate(range(pred_acts.shape[1]))}
    writer.add_scalars('auprc/{}'.format(train_or_test), aucpr_dict, iteration)

    # compute and write out accuracy every epoch (at least one correct)
    acc_one = np.mean(true_labs[np.arange(len(pred_acts)), np.argmax(pred_acts, axis=1)])
    writer.add_scalar('acc_one/{}'.format(train_or_test), acc_one, iteration)

    # compute and write out accuracy every epoch (all correct)
    worst_good_pred = np.ma.min(np.ma.masked_array(pred_acts, mask=true_labs==0), axis=1)
    best_bad_pred = np.ma.max(np.ma.masked_array(pred_acts, mask=true_labs==1), axis=1)
    acc_all = np.mean(worst_good_pred > best_bad_pred)
    writer.add_scalar('acc_all/{}'.format(train_or_test), acc_all, iteration)    
    
