import torch
import torch.nn as nn
from torchvision import transforms
from algorithms.base import Trainer
from utils import accuracy


class BaselineTrainer(Trainer):
    def __init__(self, config):
        super(BaselineTrainer, self).__init__(config)

    def get_transformers(self, config):
        transformers = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        }
        return transformers

    def get_criterion(self, config):
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), config.lr, weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)


if __name__ == '__main__':
    trainer = BaselineTrainer()
    # print(trainer.model)
    trainer.train()
