import os
from markdown import markdown
import numpy as np

import torch
import torch.nn as nn

from .args import args


def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.beckends.cudnn.enabled = True

# Save Model 

def save_model(model, pth='./src/models/trained_models.py/'):
    pth = os.path.join(pth, args.dataset)
    if not os.path.exists(pth):
        os.makedirs(pth)


    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), os.path.join(pth, args.model + '.pt'))
    print('\nModel saved!')
    return


def logging(epoch, train_stats, val_stats, writer):
    if writer is not None:
        for stat in train_stats:
            writer.add_scalar('Train/' + stat, train_stats[stat], epoch)
        
        for stat in val_stats:
            writer.add_scalar('Val/' + stat, val_stats[stat], epoch)

    print('Epoch [{:4d}/{:4d}] | Train Accuracy: {:4.2f} | Val Accuracy: {:4.2f}'.format(
        epoch, args.epochs, train_stats['accuracy'], val_stats['accuracy']))
    


def namespace2markdown(args):
    txt = '<table> <thead> <tr> <td> <strong> Hyperparameter </strong> </td> <td> <strong> Values </strong> </td> </tr> </thead>'
    txt += ' <tbody> '
    for name, var in vars(args).items():
        txt += '<tr> <td> <code>' + str(name) + ' </code> </td>' + str(var) + ' </code> </td>' + '</tr> '
    txt += '</tbody> </table>'
    return markdown(txt)


if __name__ == "__main__":
    pass