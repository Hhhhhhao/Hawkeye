
import os

def write_jobs(sku='G1', target_service='sing', target_name='msroctows', sla_tier='premium',
               model_name='baseline_vit_base_in21k_224', datasets=['aircraft', 'car', 'cub', 'dog', 'web_aircraft', 'web_bird', 'web_car'], num_seeds=3):
    description = f'{model_name} tune tasks'

    # write environment
    if 'A100' in target_service:
        image = ' amlt-sing/pytorch-1.9.0-a100'
    else:
        image = ' amlt-sing/pytorch-1.8.0' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install pyyaml\n' +\
            '  - pip install -U "ray[default]"\n' +\
            '  - pip install tensorboard\n' +\
            '  - pip install -U scikit-learn\n' +\
            '  - pip install matplotlib\n' +\
            '  - pip install progress\n' + \
            '  - pip install pandas\n' +\
            '  - pip install tensorboardx\n' +\
            '  - pip install tqdm\n' +\
            '  - pip install yacs\n' +\
            '  - python -m pip install -U scikit-image\n' +\
            '  - pip install timm\n' +\
            '  - pip install tqdm\n' +\
            '  - pip install --upgrade ruamel.yaml --ignore-installed ruamel.yaml \n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install hyperopt\n' +\
            "  - pip install 'protobuf<=3.20.1' --force-reinstall\n" +\
            '  - pwd \n'
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'{model_name}'
    filepath = 'hawkeye_fgvc'
    storage_account_name = 'ussclowpriv100data'
    container_name = 'jindwang'
    os.makedirs(f'./amlt_config', exist_ok=True)
    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        if target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')

    

        config_files = sorted(os.listdir(os.path.join('configs', model_name)))
        jobs_list = []
        for i in range(0, len(config_files), num_seeds):
            command_list = []

            for file in config_files[i:i+num_seeds]:

                config_path = os.path.join('/tmp/code/configs', model_name, file)
                command = f"python train.py --config {config_path}"
                command_list.append(command)

            if 'web' in file:
                job_name = '_'.join(file.split('_')[-5:-3])
            else:
                job_name = file.split('_')[-4]

            if job_name not in datasets:
                continue

            job_name = f'{model_name}_{job_name}'
            jobs_list.append((job_name, command_list))
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command_list = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            for command in command_list:
                w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


if __name__ == '__main__':

    write_jobs(sku='NDv2g1:16G1-V100', target_service='sing', target_name='msroctovc', model_name='baseline_resnet50_224')
    write_jobs(sku='NCv2:16G1-P100', target_service='sing', target_name='msrresrchvc', model_name='baseline_resnet50_in21k_224')
    write_jobs(sku='NCv2:16G1-P100', target_service='sing', target_name='msrresrchvc', model_name='baseline_vit_small_p16_224')
    write_jobs(sku='NDv2g1:16G1-V100', target_service='sing', target_name='msrresrchvc', model_name='baseline_vit_base_p16_in21k_224')
    write_jobs(sku='NDv2g1:16G1-V100', target_service='sing', target_name='msrresrchvc', model_name='baseline_swin_base_p4_w7_in21k_224')

