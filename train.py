

import os
import torch 
import utils

from algorithms.base import Trainer

def main():
    args = utils.setup_config()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    # TODO: add prefetcher
    # args.prefetcher = not args.no_prefetcher

    # init dist
    utils.init_distributed_device(args)

    # set random seed
    utils.set_random_seed(args.experiment.seed, args.rank)

    # create trainer
    trainer = Trainer(args)

    # start training 
    trainer.train()


if __name__ == '__main__':
    main()