import argparse
import json
import os
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set seed = {seed}')


def dump_args(args, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f)


def load_args(filepath):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(filepath, encoding='utf-8') as f:
        args.__dict__ = json.load(f)
    return args
