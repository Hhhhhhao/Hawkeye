import os
import re
import json
import numpy as np
from glob import glob
import subprocess


def get_param(file_name):

    re_lr = r"lr': (([0-9]|\.|-|e)*)"
    re_wd = r"weight_decay': (([0-9]|\.|-|e)*)"
    re_auto_aug = r"auto_aug': ((\w|')*),"
    re_rand_e = r"rand_erase': (([0-9]|\.|e|-)*)"


    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('Best trial config'):
                continue

            lr = re.search(re_lr,line).group(1)
            weight_decay = re.search(re_wd,line).group(1)
            auto_aug = re.search(re_auto_aug,line).group(1)
            if auto_aug == "None":
                auto_aug = None
            else:
                auto_aug = auto_aug[1:-1]
            rand_erase = re.search(re_rand_e,line).group(1)

    
    param = {
        'lr': round(float(lr), 6),
        'weight_decay': round(float(weight_decay), 6),
        'auto_aug': auto_aug,
        'rand_erase': round(float(rand_erase), 4)
    }
    return param


exp_name_list = ['baseline_resnet50_224', 'baseline_resnet50_in21k_224', 'baseline_vit_small_p16_224', 'baseline_vit_base_p16_in21k_224', 'baseline_swin_base_p4_w7_in21k_224']
for exp_name in exp_name_list:
    # exp_name = 'baseline_resnet50_224'
    # exp_name = 'baseline_resnet50_in21k_224'
    # exp_name = 'baseline_vit_small_p16_224'
    amlt_log_dir = f'/media/Zeus/haoc/hawkeye_fgvc/amlt/{exp_name}_tune'
    log_files = sorted(glob(os.path.join(amlt_log_dir, '*', 'stdout.txt')))
    param_dict = {}
    for log_file in log_files:
        log_param = get_param(log_file)
        log_key = log_file.split('/')[-2]
        if 'web' in log_key:
            log_key = '_'.join(log_key.split('_')[-3:-1])
        else:
            log_key = log_key.split('_')[-2]
        param_dict[log_key] = log_param

    with open(f'/media/Zeus/haoc/hawkeye_fgvc/tuned_parameteres/{exp_name}.json', 'w') as f:
        json.dump(param_dict, f, indent=4)
    
subprocess.call('python scripts/gen_config.py', shell=True)