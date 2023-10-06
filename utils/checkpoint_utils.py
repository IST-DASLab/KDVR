import os
import errno
import torch
import shutil


__all__ = ['save_checkpoint', 'load_checkpoint']


def save_checkpoint(epoch, model, checkpoint_path: str, is_best=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states.
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_state_dict'] = model.state_dict()

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}.pth')
    path_best = os.path.join(checkpoint_path, 'best_checkpoint.pth')
    path_last = os.path.join(checkpoint_path, 'last_checkpoint.pth')
    torch.save(checkpoint_dict, path_last)
    if is_best:
        print("Saving best checkpoint.")
        shutil.copyfile(path_last, path_best)
    if is_scheduled_checkpoint:
        print("Saving checkpoint on schedule.")
        shutil.copyfile(path_last, path_regular)


def load_checkpoint(full_checkpoint_path: str, model):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_dict = torch.load(full_checkpoint_path, map_location='cpu')
    except FileNotFoundError:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))

    model.load_state_dict(checkpoint_dict['model_state_dict'])
    return model


