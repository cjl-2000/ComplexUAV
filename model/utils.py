import math
import os
import shutil
import sys
import random
import errno
import time
import torch
import numpy as np
from datetime import timedelta


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60

    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)

    return '{}:{}'.format(minutes, seconds_remaining)


def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))


def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining

    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
        sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))


def adjust_learning_rate(optimizer, epoch, lr=None, total_epoch=None):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_cls(optimizer, epoch, lr=None, total_epoch=None):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epoch))
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    optimizer.param_groups[1]['lr'] = 0.5 * lr
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_path=None):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path,
                                                                        'model_best_{}_{:.4f}.pth.tar'.format(
                                                                            state["epoch"], state["best_acc"])))


def set_up_system(args):
    seed = args.seed
    if seed is not None:
        random.seed(seed + args.rank)
        np.random.seed(seed + args.rank)
        torch.manual_seed(seed + args.rank)
        torch.cuda.manual_seed(seed + args.rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
