import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw


def torch_to_PIL_single_image(x):
    x = x.numpy().transpose(1,2,0)
    for i,c in enumerate(tuple('RGB')):
        x[:,:,i] -= np.min(x[:,:,i])
        if np.max(x[:,:,i]) > 0:
            x[:,:,i] /= np.max(x[:,:,i])
            x[:,:,i] *= 255
    return Image.fromarray(x.astype(np.uint8))


def ims_labels_to_grid(ims_labels, boring_color=(255,255,255), not_boring_color=(0,255,255), ncol=8):

    img_grid_list = []
    for i, (im,label) in enumerate(ims_labels):
        img = transforms.ToPILImage()(im)
        d = ImageDraw.Draw(img)

        boring = label==0

        if not boring:
            d.text((10,10), "true: {}".format(label), fill=not_boring_color)
        else:
            d.text((10,10), "true: {}".format(label), fill=boring_color)

        img = transforms.ToTensor()(img)
        img_grid_list += [img]

    return torch_to_PIL_single_image(make_grid(img_grid_list, nrow=ncol))

def ims_preds_to_grid(ims_preds, boring_color=(255,255,255), not_boring_color=(0,255,255), ncol=8):

    img_grid_list = []
    for i, (im,pred) in enumerate(ims_preds):
        img = transforms.ToPILImage()(im)
        d = ImageDraw.Draw(img)

        boring = pred==0

        if not boring:
            d.text((10,10), "pred: {}".format(pred), fill=not_boring_color)
        else:
            d.text((10,10), "pred: {}".format(pred), fill=boring_color)

        img = transforms.ToTensor()(img)
        img_grid_list += [img]

    return torch_to_PIL_single_image(make_grid(img_grid_list, nrow=ncol))


def ims_labels_preds_to_grid(ims_labels_preds, boring_color=(255,255,255), right_color=(0,255,255), wrong_color=(255,0,0), ncol=8):

    img_grid_list = []
    for i, (im,label,pred) in enumerate(ims_labels_preds):
        img = transforms.ToPILImage()(im)
        d = ImageDraw.Draw(img)

        boring = label==0
        right = label==pred
        wrong = label!=pred

        if right and not boring:
            d.text((10,10), "true: {}".format(label), fill=right_color)
            d.text((10,20), "pred: {}".format(pred), fill=right_color)
        elif wrong:
            d.text((10,10), "true: {}".format(label), fill=wrong_color)
            d.text((10,20), "pred: {}".format(pred), fill=wrong_color)
        else:
            d.text((10,10), "true: {}".format(label), fill=boring_color)
            d.text((10,20), "pred: {}".format(pred), fill=boring_color)

        img = transforms.ToTensor()(img)
        img_grid_list += [img]

    return torch_to_PIL_single_image(make_grid(img_grid_list, nrow=ncol))

