import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath('.'))
from algorithms.base import Trainer
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy, Timer, NativeScaler, reduce_tensor


class PIMTrainer(Trainer):
    def __init__(self):
        super(PIMTrainer, self).__init__()

    def batch_training(self, data):
        # images, labels = self.to_device(data['img']), self.to_device(data['label'])
        images = data['img'].to(self.device)
        labels = data['label'].to(self.device)
        batch_size = images.size(0)

        # forward
        with self.amp_autocast():
            outs = self.model(images)
            # loss = self.criterion(outputs, labels)

            loss = 0.
            for name in outs:
                
                if "select_" in name:
                    if not self.config.model.pim.use_selection:
                        raise ValueError("Selector not use here.")
                    if self.config.model.pim.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, self.config.model.num_classes).contiguous()
                        loss_s = F.cross_entropy(logit, 
                                                 labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += self.config.model.pim.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not self.config.model.pim.use_selection:
                        raise ValueError("Selector not use here.")

                    if self.config.model.pim.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, self.config.model.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, self.config.model.num_classes]) - 1
                        labels_0 = labels_0.to(images.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += self.config.model.pim.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not self.config.model.pim.use_fpn:
                        raise ValueError("FPN not use here.")
                    if self.config.model.pim.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = F.cross_entropy(outs[name].mean(1), labels)
                        loss += self.config.model.pim.lambda_b * loss_b
                    else:
                        loss_b = 0.0
                
                elif "comb_outs" in name:
                    if not self.config.model.pim.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if self.config.model.pim.lambda_c != 0:
                        loss_c = F.cross_entropy(outs[name], labels)
                        loss += self.config.model.pim.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

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
        acc = accuracy(outs['comb_outs'], labels, 1)
        if self.distributed:
            loss = reduce_tensor(loss.data, self.config.world_size)
            acc = reduce_tensor(acc.data, self.config.world_size)

        self.average_meters['acc'].update(acc.item(), images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        # images, labels = self.to_device(data['img']), self.to_device(data['label'])
        images = data['img'].to(self.device)
        labels = data['label'].to(self.device)
        with self.amp_autocast():
            outs = self.model(images)
        acc = accuracy(outs['comb_outs'], labels, 1)
        if self.distributed:
            acc = reduce_tensor(acc.data, self.config.world_size)
        self.average_meters['acc'].update(acc.item(), images.size(0))


if __name__ == '__main__':
    trainer = PIMTrainer()
    # print(trainer.model)
    trainer.train()
