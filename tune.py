

import os
import torch 
import utils
from functools import partial
from algorithms.base import Trainer
from yacs.config import CfgNode as CN

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch



def tune_hyperparameters(args, config, change_exp_name=True):
    args.defrost()

    if change_exp_name:
        args.experiment.name = args.experiment.name + '_{}'.format(session.get_trial_id())

    args.train.optimizer.lr = config["lr"]

    args.train.optimizer.weight_decay = config["weight_decay"]

    args.dataset.transformer.auto_aug = config["auto_aug"]

    args.dataset.transformer.rand_erase = config["rand_erase"]

    args.freeze()
    return args 



def objective(config, args):    


    # TODO: add prefetcher
    # args.prefetcher = not args.no_prefetcher

    # init dist
    utils.init_distributed_device(args)

    # set random seed
    utils.set_random_seed(args.experiment.seed, args.rank)

    # tune_hyperparameters
    args = tune_hyperparameters(args, config)

    # create trainer
    trainer = Trainer(args)

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            trainer.load_checkpoint(os.path.join(loaded_checkpoint_dir, 'last_model.pth'))

    # start training 
    acc = trainer.train()
    return {'accuracy': acc}




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/media/Dormammu/haoc/hawkeye_fgvc/configs/baseline_r50_in1k_cub_224_b32.yaml', help="config file")
    parser.add_argument('--max_t', type=int, default=30, help="maximum epochs for each tune trail")
    parser.add_argument('--n_cpu', type=float, default=4, help="number of cpus for  each trail")
    parser.add_argument('--n_gpu', type=float, default=1, help="number of gpus for  each trail")
    parser.add_argument('--n_trials', type=int, default=1, help="number of trails for  tuning")
    args = parser.parse_args()

    with open(args.config) as f:
        config_args = CN.load_cfg(f)
    # args.experiment.debug = True
    config_args.experiment.tune = True
    config_args.experiment.log_dir += '_tune'
    config_args.freeze()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    # define search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "auto_aug": tune.choice(["ta_wide", "rand_aug", 'auto_aug', None]),
        "rand_erase": tune.uniform(0, 0.25),
    }
    current_best_params = [{
    'lr': 1e-4,
    'weight_decay': 2e-3,
    'auto_aug': "ta_wide",
    'rand_erase': 0.2,
    }]
    hyperopt_search = HyperOptSearch(
        metric="accuracy", mode="max",
        points_to_evaluate=current_best_params)
    scheduler = ASHAScheduler(
        max_t=args.max_t,
        grace_period=5,
        reduction_factor=2,
        metric="accuracy", mode="max")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(objective, args=config_args)),
            resources={"cpu": args.n_cpu, "gpu": args.n_gpu}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=hyperopt_search,
            num_samples=args.n_trials,
        ),
        run_config=ray.air.config.RunConfig(
            local_dir='/media/Zeus/ray_results'
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    
    tune_hyperparameters(config_args, best_result.config, change_exp_name=False)
    with open(args.config, 'w') as f:
        f.write(config_args.dump())

    

    
