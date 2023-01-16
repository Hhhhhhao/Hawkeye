import os
import sys
import torch
from contextlib import suppress
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DataParallel, DistributedDataParallel 
from tqdm import tqdm
import logging
from shutil import copyfile

import utils
from dataset.dataset import FGDataset
from dataset.transforms import ClassificationPresetTrain, ClassificationPresetEval
from model.registry import MODEL
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy, Timer, NativeScaler, reduce_tensor


def emergency_save(func):
    """ Save checkpoint when `KeyboardInterrupt` or other errors occur.
    """

    def _emergency_save(self):
        try:
            func(self)
        except KeyboardInterrupt:
            self.logger.info('KeyboardInterrupt - try to save checkpoint ...')
            self.save_checkpoint()
        except Exception as e:
            import traceback
            self.logger.error(repr(e))
            self.logger.error(traceback.format_exc())
            self.logger.info('try to save checkpoint ...')
            self.save_checkpoint()

    return _emergency_save


class Trainer(object):
    """Base trainer
    """

    def __init__(self, config):
        # self.config = setup_config()
        self.config = config

        # set epoch, resume flag and log_root
        self.device = torch.device(self.config.device)
        self.distributed = config.distributed
        self.epoch = 0
        self.start_epoch = 0
        self.total_epoch = self.config.train.epoch
        self.resume = 'resume' in self.config.experiment and self.config.experiment.resume
        self.debug = self.config.experiment.debug if 'debug' in self.config.experiment else False
        self.log_root = os.path.join(self.config.experiment.log_dir, self.config.experiment.name)
        self.report_one_line = True  # logger report acc and loss in one line when training

        # log root directory should not already exist
        if not self.resume and not self.debug and utils.is_primary(config):
            assert not os.path.exists(self.log_root), 'Experiment log folder already exists!!'
            # create log root directory and copy
            os.makedirs(self.log_root)
            print(f'Created log directory: {self.log_root}')
            # copy yaml file and train.py
            with open(os.path.join(self.log_root, 'train_config.yaml'), 'w') as f:
                f.write(self.config.__str__())
            copyfile(sys.argv[0], os.path.join(self.log_root, 'train.py'))

        # logger and tensorboard writer
        self.logger = self.get_logger()
        self.tb_writer = SummaryWriter(self.log_root)
        self.logger.info(f'Train Config:\n{self.config.__str__()}')

        # build dataloader and model
        self.transformers = self.get_transformers(self.config.dataset.transformer)
        self.collate_fn = self.get_collate_fn()
        self.datasets = self.get_dataset(self.config.dataset)
        self.dataloaders = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.model_to_device(self.model)
        self.logger.info(f'Building model {self.config.model.name} OK!')
    
        # build loss, optimizer, scheduler
        self.criterion = self.get_criterion(self.config.train.criterion)
        self.optimizer = self.get_optimizer(self.config.train.optimizer)
        self.scheduler = self.get_scheduler(self.config.train.scheduler)
        self.clip_grad = self.config.train.optimizer.clip_grad

        # check amp
        self.amp_autocast, self.loss_scaler = self.get_amp(config)

        # resume from checkpoint
        if self.resume:
            self.logger.info(f'Resuming from `{self.resume}`')
            self.load_checkpoint(self.config.experiment.resume)

        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

        # timer
        self.timer = Timer()

        self.logger.info('Training Preparation Done!')

    def __del__(self):
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

    def get_logger(self):
        # TODO: disable non primary
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        if utils.is_primary(self.config):
            logger.setLevel(logging.INFO)

            screen_handler = TqdmHandler()
            screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
            logger.addHandler(screen_handler)

            complicated_format = logging.Formatter('%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s \
                                %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
            simple_format = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

            file_handler = logging.FileHandler(os.path.join(self.log_root, 'report.log'), encoding='utf8')
            file_handler.setFormatter(simple_format)
            logger.addHandler(file_handler)
        return logger

    def get_performance_meters(self):
        return {
            'train': {
                metric: PerformanceMeter(higher_is_better=False if 'loss' in metric else True)
                for metric in ['acc', 'loss']
            },
            'val': {
                metric: PerformanceMeter() for metric in ['acc']
            },
            'val_first': {
                metric: PerformanceMeter() for metric in ['acc']
            }
        }

    def get_average_meters(self):
        meters = ['acc', 'loss']  # Reset every epoch. 'acc' is reused in train/val/val_first stage.
        return {
            meter: AverageMeter() for meter in meters
        }

    def reset_average_meters(self):
        for meter in self.average_meters:
            self.average_meters[meter].reset()

    def get_model(self, config):
        """Build and load model in config
        """
        name = config.name
        model = MODEL.get(name)(config)

        if 'load' in config and config.load != '':
            self.logger.info(f'Loading model from {config.load}')
            state_dict = torch.load(config.load, map_location='cpu')
            model.load_state_dict(state_dict)
            self.logger.info(f'OK! Model loaded from {config.load}')
        return model

    def get_transformers(self, config):
        transformers = {
            'train': ClassificationPresetTrain(
                crop_size=config['image_size'],
                auto_augment_policy=config['auto_aug'], #"ta_wide",
                random_erase_prob=config['rand_erase'],
            ),
            'val': ClassificationPresetEval(
                crop_size=config['image_size'],
                resize_size=config['resize_size']
            )
        }
        return transformers

    def get_collate_fn(self):
        return {
            'train': None,
            'val': None
        }

    def get_dataset(self, config):
        if self.config.rank != 0 and self.distributed:
            torch.distributed.barrier()
        splits = ['train', 'val']
        meta_paths = {
            split: os.path.join(config.meta_dir, split + '.txt') for split in splits
        }
        datasets = {
            split: FGDataset(config.root_dir, meta_paths[split], transform=self.transformers[split]) for split in splits
        }
        if self.config.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return datasets

    def get_dataloader(self, config):
        # TODO: add distributed sampler
        splits = ['train', 'val']
        if self.distributed:
            dataloaders = {
                split: DataLoader(
                    self.datasets[split],
                    config.batch_size, num_workers=config.num_workers, pin_memory=True,
                    collate_fn=self.collate_fn[split],
                    sampler=DistributedSampler(self.datasets[split]) if split == 'train' else None
                ) for split in splits
            }
        else:
            dataloaders = {
                split: DataLoader(
                    self.datasets[split],
                    config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=split == 'train',
                    collate_fn=self.collate_fn[split],
                ) for split in splits
            }
        return dataloaders

    def get_criterion(self, config):
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # TODO: change here
    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # TODO: change here
    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)
    
    def get_amp(self, config):
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if config.experiment.use_amp:
            amp_autocast = partial(torch.autocast, device_type=self.device.type)
            if self.device.type == 'cuda':
                loss_scaler = NativeScaler()
                self.logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            self.logger.info('AMP not enabled. Training in float32.')
        return amp_autocast, loss_scaler

    
    def model_to_device(self, model):
        if self.distributed:
            self.logger.info("Training in distributed mode with multiple processes.")
            self.logger.info(f"Process {self.config.rank}, total {self.config.world_size}, device {self.config.device}.")
            model = model.to(self.device)
            model = DistributedDataParallel(model, device_ids=[self.device], broadcast_buffers=True)
        else:
            if self.config.world_size > 1:
                self.logger.info("Training in data parallel mode with single process.")
                model = DataParallel(model)
                model = model.to(self.device)
            else:
                self.logger.info("Training in single gpu")
                model = model.to(self.device)
        return model


    def get_model_module(self, model=None):
        """get `model` in single-gpu mode or `model.module` in multi-gpu mode.
        """
        if model is None:
            model = self.model
        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            return model.module
        else:
            return model

    @emergency_save
    def train(self):
        config = self.config.train  # local config for training stage

        # validate firstly
        if 'val_first' in config and config.val_first:
            self.logger.info('Validate model before training.')
            self.validate()
            self.performance_meters['val_first']['acc'].update(self.average_meters['acc'].avg)
            self.report(epoch=0, split='val_first')

        self.model.train()

        for epoch in range(self.start_epoch, self.total_epoch):
            self.epoch = epoch
            self.reset_average_meters()
            self._on_start_epoch()

            # distributed sampler
            if self.distributed and hasattr(self.dataloaders['train'].sampler, 'set_epoch'):
                self.dataloaders['train'].sampler.set_epoch(epoch)

            # train stage
            self.logger.info(f'Starting epoch {epoch + 1} ...')
            self.timer.tick()
            training_bar = tqdm(self.dataloaders['train'], ncols=100, disable=not utils.is_primary(self.config))
            for data in training_bar:
                self._on_start_forward()
                self.batch_training(data)
                torch.cuda.synchronize()
                self._on_end_forward()
                training_bar.set_description(f'Train Epoch [{self.epoch + 1}/{self.total_epoch}]')
                training_bar.set_postfix(acc=self.average_meters['acc'].avg, loss=self.average_meters['loss'].avg)
            duration = self.timer.tick()
            self.logger.info(f'Training duration {duration:.2f}s!')

            # train stage metrics
            self.update_performance_meter('train')
            self.report(epoch=epoch + 1, split='train')

            # val stage
            self.logger.info(f'Starting validation stage in epoch {epoch + 1} ...')
            self.timer.tick()
            # validate
            self.validate()
            duration = self.timer.tick()
            self.logger.info(f'Validation duration {duration:.2f}s!')

            # val stage metrics
            val_acc = self.average_meters['acc'].avg
            if self.performance_meters['val']['acc'].best_value is not None:
                is_best = epoch >= 5 and val_acc > self.performance_meters['val']['acc'].best_value
            else:
                is_best = epoch >= 5
            self.update_performance_meter('val')
            self.report(epoch=epoch + 1, split='val')

            self.do_scheduler_step()
            self.logger.info(f'Epoch {epoch + 1} Done!')

            # save model
            if epoch != 0 and (epoch + 1) % config.save_frequence == 0 and utils.is_primary(self.config):
                self.logger.info('Saving model ...')
                self.save_model()
                # self.logger.info('Saving checkpoint ...')
                # self.save_checkpoint()
            if is_best and utils.is_primary(self.config):
                self.logger.info('Saving best model ...')
                self.save_model('best_model.pth')

            # hook: on_end_epoch
            self._on_end_epoch()

        self.logger.info(f'best acc:{self.performance_meters["val"]["acc"].best_value}')

    def batch_training(self, data):
        # images, labels = self.to_device(data['img']), self.to_device(data['label'])
        images = data['img'].to(self.device)
        labels = data['label'].to(self.device)

        # forward
        with self.amp_autocast():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        # backward
        self.optimizer.zero_grad()
        if self.loss_scaler is not None:
            self.loss_scaler(
                loss, self.optimizer,
                clip_grad=self.config.train.optimizer.clip_grad,
                parameters=self.model.parameters()),
        else:
            loss.backward()
            if self.config.train.optimizer.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.optimizer.clip_grad, norm_type=2.0)
            self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(outputs, labels, 1)
        if self.distributed:
            loss = reduce_tensor(loss.data, self.config.world_size)
            acc = reduce_tensor(acc.data, self.config.world_size)

        self.average_meters['acc'].update(acc.item(), images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))


    def validate(self):
        self.model.train(False)
        self.reset_average_meters()

        with torch.no_grad():
            val_bar = tqdm(self.dataloaders['val'], ncols=100)
            for data in val_bar:
                self.batch_validate(data)
                val_bar.set_description(f'Val Epoch [{self.epoch + 1}/{self.total_epoch}]')
                val_bar.set_postfix(acc=self.average_meters['acc'].avg)

        self.model.train(True)

    def batch_validate(self, data):
        # images, labels = self.to_device(data['img']), self.to_device(data['label'])
        images = data['img'].to(self.device)
        labels = data['label'].to(self.device)
        with self.amp_autocast():
            logits = self.model(images)
        acc = accuracy(logits, labels, 1)
        if self.distributed:
            acc = reduce_tensor(acc.data, self.config.world_size)
        self.average_meters['acc'].update(acc.item(), images.size(0))

    def do_scheduler_step(self):
        self.scheduler.step()

    def update_performance_meter(self, split):
        if split == 'train':
            self.performance_meters['train']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['train']['loss'].update(self.average_meters['loss'].avg)
        elif split == 'val':
            self.performance_meters['val']['acc'].update(self.average_meters['acc'].avg)

    def report(self, epoch, split='train'):
        # tensorboard summary-writer and logger
        for metric in self.performance_meters[split]:
            value = self.performance_meters[split][metric].current_value
            self.tb_writer.add_scalar(f'{split}/{metric}', value, global_step=epoch)
            if not self.report_one_line:
                self.logger.info(f'Epoch:{epoch}\t{split}/{metric}: {value}')
        if self.report_one_line:
            metric_str = '  '.join([f'{metric}: {self.performance_meters[split][metric].current_value:.2f}'
                                    for metric in self.performance_meters[split]])
            self.logger.info(f'Epoch:{epoch}\t{metric_str}')

    def save_model(self, name=None):
        model_name = self.config.model.name
        if name is None:
            path = os.path.join(self.log_root, f'{model_name}_epoch_{self.epoch + 1}.pth')
        else:
            path = os.path.join(self.log_root, name)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'model saved to: {path}')

    def save_checkpoint(self):
        path = os.path.join(self.log_root, f'checkpoint_epoch_{self.epoch}.pth')
        checkpoint = {
            'epoch': self.epoch,
            'model': self.get_model_module().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if self.loss_scaler is not None:
            checkpoint['loss_scaler'] = self.loss_scaler.state_dict()
        torch.save(checkpoint, path)
        self.logger.info(f'checkpoint successfully saved to: {path}')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.loss_scaler is not None and 'loss_sclaer' in checkpoint:
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.logger.info(f'load checkpoint from: {path}, start_epoch: {checkpoint["epoch"]}')

    # hooks used in trainer
    def _on_start_epoch(self):
        if 'hook' in self.config and 'on_start_epoch' in self.config.hook:
            return self.on_start_epoch(self.config.hook.on_start_epoch)
        else:
            return self.on_start_epoch(None)

    def _on_end_epoch(self):
        if 'hook' in self.config and 'on_end_epoch' in self.config.hook:
            return self.on_end_epoch(self.config.hook.on_end_epoch)
        else:
            return self.on_end_epoch(None)

    def _on_start_forward(self):
        if 'hook' in self.config and 'on_start_forward' in self.config.hook:
            return self.on_start_forward(self.config.hook.on_start_forward)
        else:
            return self.on_start_forward(None)

    def _on_end_forward(self):
        if 'hook' in self.config and 'on_end_forward' in self.config.hook:
            return self.on_end_forward(self.config.hook.on_end_forward)
        else:
            return self.on_end_forward(None)

    # hooks to implement
    def on_start_epoch(self, config):
        lrs_str = "  ".join([f'{p["lr"]}' for p in self.optimizer.param_groups])
        self.logger.info(f'Epoch:{self.epoch + 1}  LR: {lrs_str}')

    def on_end_epoch(self, config):
        pass

    def on_start_forward(self, config):
        pass

    def on_end_forward(self, config):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
