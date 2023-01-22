import os
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url, HASH_REGEX, download_url_to_file, urlparse
try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir


def initialize_weights(m) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, val=0)


def initialize_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


def load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {}
    unmatched_key = []
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict[k] = v
        else:
            unmatched_key.append(k)
    # pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("unmatched keys: {}".format(unmatched_key))


def freeze(m):
    if isinstance(m, nn.Module):
        for param in m.parameters():
            param.require_grad = False
    elif isinstance(m, list):
        for param in m:
            param.require_grad = False
    else:
        raise NotImplementedError()


def unfreeze(m):
    if isinstance(m, nn.Module):
        for param in m.parameters():
            param.require_grad = True
    elif isinstance(m, list):
        for param in m:
            param.require_grad = True
    else:
        raise NotImplementedError()


def get_cache_dir(child_dir=''):
    """
    Returns the location of the directory where models are cached (and creates it if necessary).
    """
    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def download_cached_file(url, check_hash=True, progress=False):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(get_cache_dir(), filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file