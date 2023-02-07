import os
import re
import json
import numpy as np
from glob import glob
import subprocess


def get_acc(file_name):

    re_acc = r"best acc:(([0-9]|\.|-|e)*)"

    acc_list =[]
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'best acc' in line:
                acc = re.search(re_acc,line).group(1)
                acc = float(acc)
                acc_list.append(acc)
    return acc_list


exp_name_list = ['baseline_resnet50_224', 'baseline_resnet50_in21k_224', 'baseline_vit_small_p16_224']
all_acc_dict = {}
for exp_name in exp_name_list:
    # exp_name = 'baseline_resnet50_224'
    # exp_name = 'baseline_resnet50_in21k_224'
    # exp_name = 'baseline_vit_small_p16_224'
    amlt_log_dir = f'/media/Zeus/haoc/hawkeye_fgvc/amlt/{exp_name}'
    log_files = sorted(glob(os.path.join(amlt_log_dir, '*', 'stdout.txt')))
    acc_dict = {}

    for log_file in log_files:
        acc_list = get_acc(log_file)
        log_key = log_file.split('/')[-2]
        if 'web' in log_key:
            log_key = '_'.join(log_key.split('_')[-2:])
        else:
            log_key = log_key.split('_')[-1]

        acc_dict[log_key] = acc_list

    all_acc_dict[exp_name] = acc_dict


import xlwt

datasets = all_acc_dict['baseline_resnet50_224'].keys()
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('acc',cell_overwrite_ok=True)
for j, dataset in enumerate(datasets):
    worksheet.write(0, j + 1, dataset)

for i, exp_name in enumerate(exp_name_list):
    worksheet.write(i + 1, 0, exp_name)
    for j, dataset in enumerate(datasets):
        acc_list = all_acc_dict[exp_name][dataset]
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)

        worksheet.write(i + 1, j + 1, str(round(mean_acc, 2))+u"\u00B1"+str(round(std_acc, 2)))

workbook.save('./results/final_res.xls')