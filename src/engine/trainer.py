import motmetrics
import os
import time
import torch
import numpy as np

from torch.cuda.amp import autocast, GradScaler
from progress.bar import Bar

from src.utils.data_parallel import DataParallel

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, images, targets, mode='train'):
        outputs = self.model(images, mode)
        if mode == 'train'
            loss, loss_stats = self.loss(outputs, targets)
            return outputs, loss, loss_stats
        else:
            return outputs

class Trainer(object):
    def __init__(self, opt, model, optimizer, loss):
        self.opt = opt
        self.amp_enabled = getattr(opt, 'amp', False)
        self.scaler = GradScaler() if self.amp_enabled else None
        self.model_with_loss = ModelWithLoss(model, loss)
        self.optimizer = optimizer

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train(self, train_loader, epoch ,logger):
        self.model_with_loss.train()
        num_iters = len(train_loader)
        start_time = time.time()
        bar = Bar(f'Epoch {epoch + 1, self.opt.num_epochs }', max=num_iters, fill='▇',
                suffix='%(percent)3d%% | ETA: %(etamsg)7s | loss: %(loss)8.4f | hm: %(hm)8.4f | wh: %(wh)8.4f | off: %(off)8.4f | track: %(track)8.4f | seq: %(seq)8.4f')

        # 遍历数据集
        for iter_id, (images, targets) in enumerate(train_loader):
            if iter_id > num_iters:
                break
            if self.amp_enabled:
                with autocast():
                    outputs, loss, loss_stats = self.model_with_loss(images, targets, mode)
                loss = loss.mean()
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, loss, loss_stats = self.model_with_loss(images, targets, mode)
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            bar.loss = float(loss.mean().cpu().detach().numpy())
            bar.hm = float(loss_stats['hm_loss'].mean().cpu().detach().numpy())
            bar.wh = float(loss_stats['wh_loss'].mean().cpu().detach().numpy())
            bar.off = float(loss_stats['off_loss'].mean().cpu().detach().numpy())
            bar.track = float(loss_stats['track_loss'].mean().cpu().detach().numpy())
            bar.seq = float(loss_stats['seq_loss'].mean().cpu().detach().numpy())
            # ETA美化为分+秒
            eta = bar.eta
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            bar.etamsg = f'{eta_min}m{eta_sec}s'
            bar.next()

        bar.finish()

        # 记录日志
        logger.write(f"epoch: {} |")  
        for key, value in loss_stats.items():
            logger.write(f"{key}: {value.mean().item():.4f} | ")
        logger.write('\n')

        # 统计epoch耗时
        elapsed_time = time.time() - start_time
        epoch_min = int(elapsed_time // 60)
        epoch_sec = int(elapsed_time % 60)
        print(f"Epoch {epoch + 1}/{self.opt.num_epochs} time: {epoch_min}m{epoch_sec}s.")
        logger.write(f"Epoch {epoch + 1}/{self.opt.num_epochs} time: {epoch_min}m{epoch_sec}s.\n")  

        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best': best
        }, os.path.join(opt.save_dir, 'model_last.pth'))

        # 按照 lr_step 调整学习率
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def val(self, val_loader, epoch, logger, best):
        self.model_with_loss.eval()
        num_iters = len(val_loader)
        start_time = time.time()
        torch.cuda.empty_cache()

        # 如果使用多GPU，获取module
        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module

        bar = 

        for iter_id, (images, targets) in enumerate(val_loader):
            if iter_id > num_iters:
                break

            with torch.no_grad():
                outputs = self.model_with_loss(images, targets, mode='val')

            bar.next()

        bar.finish()
