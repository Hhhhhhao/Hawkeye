import os
import json
from yacs.config import CfgNode as CN




def create_config(pipeline_name='baseline', model_name='ResNet50', dataset_name='cub', data_path='CUB',
                  batch_size=32, num_workers=4, image_size=224, crop_ratio=0.875, num_classes=200,
                  epochs=30, 
                  seed=0, use_amp=False):

    cfg = CN()


    base_dir = '/media/Zeus/hawkeye_fgvc'
    # base_dir = '/tmp/code'
    save_dir = './results'
    # save_dir = '/mnt/default/projects/fgvc_2023/v1/'
    data_dir = '/media/Zeus/datasets'
    # data_dir = '/mnt/data/dataset/fgvc_datasets/datasets'

    # experiments
    cfg.experiment = CN()
    cfg.experiment.log_dir = os.path.join(save_dir, f'{pipeline_name.lower()}_{model_name.lower()}')
    cfg.experiment.name = f'{model_name.lower()}_{dataset_name.lower()}_{image_size}_{seed}'
    cfg.experiment.seed = seed
    cfg.experiment.use_amp = use_amp

    # dataset 
    cfg.dataset = CN()
    cfg.dataset.name = dataset_name
    cfg.dataset.batch_size = batch_size
    cfg.dataset.meta_dir = f'{base_dir}/metadata/{dataset_name}'
    cfg.dataset.num_workers = num_workers
    cfg.dataset.root_dir = os.path.join(data_dir, data_path)
    cfg.dataset.transformer = CN()
    cfg.dataset.transformer.auto_aug = None
    cfg.dataset.transformer.rand_erase = 0.0
    cfg.dataset.transformer.resize_size = int(image_size / crop_ratio)
    cfg.dataset.transformer.image_size = image_size

    # trianer
    cfg.trainer = CN()
    cfg.trainer.name = pipeline_name

    # model 
    cfg.model = CN()
    if pipeline_name == 'PIM':
        cfg.model.name = 'PIM'
        cfg.model.backbone_name = model_name
        cfg.model.pim = CN()
        cfg.model.pim.use_fpn = True
        cfg.model.pim.use_selection = True
        cfg.model.pim.use_combiner = True
        cfg.model.pim.fpn_size = 1536
        cfg.model.pim.proj_type = 'Linear'
        cfg.model.pim.upsample_type = 'Conv'
        cfg.model.pim.num_selects = CN({
            'layer1': 256,
            'layer2': 128,
            'layer3': 32,
            'layer4': 32
        })
        cfg.model.pim.lambda_b = 0.5
        cfg.model.pim.lambda_s = 0.0
        cfg.model.pim.lambda_n = 5.0
        cfg.model.pim.lambda_c = 1.0
    else:
        # baseline
        cfg.model.name = model_name
    cfg.model.num_classes = num_classes

    # train 
    cfg.train = CN()
    cfg.train.criterion = CN() 
    cfg.train.criterion.name = 'CrossEntropyLoss'
    cfg.train.epoch = epochs
    cfg.train.optimizer = CN()
    cfg.train.optimizer.clip_grad = None
    cfg.train.optimizer.name = 'Adam'
    cfg.train.optimizer.lr = 0.001
    cfg.train.optimizer.momentum = 0.9
    cfg.train.optimizer.weight_decay = 1.0e-4
    cfg.train.save_frequence = cfg.train.epoch
    cfg.train.scheduler = CN()
    cfg.train.scheduler.name = 'CosineAnnealingLR'
    cfg.train.scheduler.T_max = cfg.train.epoch
    cfg.train.scheduler.eta_min = 1.0e-8

    return cfg


def load_tune_param(cfg, pipeline_name, model_name, image_size, dataset_name):
    json_file = os.path.join(f'./tuned_parameteres/{pipeline_name.lower()}_{model_name.lower()}_{image_size}.json')
    if not os.path.exists(json_file):
        return cfg

    with open(json_file, 'r') as f:
        param_dict = json.load(f)[dataset_name]
    
    cfg.train.optimizer.lr = param_dict["lr"]
    cfg.train.optimizer.weight_decay = param_dict["weight_decay"]
    cfg.dataset.transformer.auto_aug = param_dict["auto_aug"]
    cfg.dataset.transformer.rand_erase = param_dict["rand_erase"]
    # check if layer_scale
    return cfg


def gen_exp(pipeline_name, model_name, image_size):

    config_dir = f'./configs/{pipeline_name.lower()}_{model_name.lower()}_{image_size}'
    os.makedirs(config_dir, exist_ok=True)

    datasets = ['cub', 'car', 'dog', 'aircraft', 'web_bird', 'web_car', 'web_aircraft']
    seeds = [42, 123, 7]

    for dataset_name in datasets:
        
        if dataset_name == 'cub':
            num_classes = 200
            data_path = 'CUB_200_2011/CUB_200_2011/images'
        elif dataset_name == 'car':
            num_classes = 196
            data_path = 'Stanford_Car'
        elif dataset_name == 'dog':
            num_classes = 120
            data_path = 'Stanford_Dog/Images'
        elif dataset_name == 'aircraft':
            num_classes = 100
            data_path = 'fgvc-aircraft-2013b/data/images'
        elif dataset_name == 'web_bird':
            num_classes = 200 
            data_path = 'web_bird'
        elif dataset_name == 'web_car':
            num_classes = 196
            data_path = 'web_car'
        elif dataset_name == 'web_aircraft':
            num_classes = 100 
            data_path = 'web_aircraft'
        


        for seed in seeds:
            cfg = create_config(
                pipeline_name=pipeline_name, model_name=model_name, dataset_name=dataset_name, data_path=data_path,
                image_size=image_size, crop_ratio=0.875, num_classes=num_classes, epochs=50, seed=seed
            )
            cfg = load_tune_param(cfg, pipeline_name=pipeline_name, model_name=model_name, image_size=image_size, dataset_name=dataset_name)
            
            save_path = os.path.join(config_dir, f'{pipeline_name.lower()}_{model_name.lower()}_{dataset_name.lower()}_{cfg.dataset.transformer.image_size}_b{cfg.dataset.batch_size}_seed{seed}.yaml')
            with open(save_path, 'w') as f:
                f.write(cfg.dump())
        

if __name__ == '__main__':

    # gen_exp('Baseline', 'ResNet50', 224)
    # gen_exp('Baseline', 'ResNet50_IN21K', 224)
    # gen_exp('Baseline', 'ViT_Small_P16', 224)
    # gen_exp('Baseline', 'ViT_Base_P16_IN21K', 224)
    # gen_exp('Baseline', 'Swin_Tiny_P4_W7_IN21K', 224)
    # gen_exp('Baseline', 'Swin_Base_P4_W7_IN21K', 224)
    # gen_exp('Baseline', 'CAFormer_S18', 224)
    # gen_exp('Baseline', 'CAFormer_S18_IN21K', 224)
    # gen_exp('Baseline', 'CAFormer_S36', 224)
    # gen_exp('Baseline', 'CAFormer_S36_IN21K', 224)
    # gen_exp('Baseline', 'CAFormer_M36', 224)


    # gen_exp('Baseline', 'TransFG_ViT_Small_P16_IN21K', 224)
    # gen_exp('Baseline', 'TransFG_ViT_Base_P16_IN21K', 224)

    gen_exp('PIM', 'Swin_Tiny_P4_W7_IN21K', 224)
    gen_exp('PIM', 'Swin_Base_P4_W7_IN21K', 224)
