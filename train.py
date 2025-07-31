import os
import time
import torch

from torch.utils.data import Dataloader

from src.data_process.dataset import RSIPACDatset
from src.engine.trainer import Trainer
from src.loss.mp2loss import MP2Loss
from src.models.model import MP2Net
from src.opts import Opts
from src.utils.tools import print_banner
from src.utils.logger import Logger


def resume(opt):
    # 断点续训
    resume_path = getattr(opt, 'model_path', None)
    if resume_path is None or not os.path.exists(resume_path):
        resume_path = os.path.join(opt.save_dir, 'model_last.pth')
    start_epoch = 0
    best = -1   # 最佳验证集 ap50
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=opt.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best = checkpoint.get('best', -1)
            print(f'Resume training from epoch {start_epoch+1} (loaded from {resume_path})')
        else:
            # 只保存了模型参数
            model.load_state_dict(checkpoint)
            print(f'Loaded model weights from {resume_path}, start training from scratch.')
            start_epoch = 0
            best = -1
    else:
        print('Start training from scratch.')
    return start_epoch, best

def train(opt):
     # ********************** 准备训练 **********************
    print_banner('Preparing training')

    torch.manual_seed(opt.seed)
    main_gpu = opt.gpus[0]
    opt.device = torch.device(f'cuda:{main_gpu}' if torch.cuda.is_available() and main_pu>=0 else 'cpu')
    print(f"Using device: {opt.device}")

    # 数据集
    train_dataset = RSIPACDatset(opt, mode='train')
    val_dataset = RSIPACDatset(opt, mode='val')
    train_loader = Dataloader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
                            num_workers=opt.num_workers,pin_memory=True, drop_last=True)
    val_loader = Dataloader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
                        
    # 模型
    model = MP2Net(opt)
    print(f"Create model: {model.__class__.__name__}")

    # 起始epoch和最佳验证集结果
    start_epoch, best = resume(opt)
    print(f"Start epoch: {start_epoch}, Best validation AP50: {best}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print(f"Create optimizer: {optimizer.__class__.__name__}")

    # 损失函数
    loss = MP2Loss(opt)
    print(f"Create loss function: {loss.__class__.__name__}")

    # 训练器
    trainer = Trainer(opt, model, optimizer, loss)
    trainer.set_device(opt.gpus, opt.device)
    print(f"Create trainer: {trainer.__class__.__name__}")

    # 保存目录
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # 记录器
    logger = Logger(opt)
    print(f"Create logger: {logger.__class__.__name__}")

    # *********************** 开始训练 **********************
    print_banner('Starting training')
    total_start_time = time.time()

    for epoch in range(start_epoch, opt.num_epochs):
        trainer.train(train_loader, epoch, logger)              # 训练

        if (epoch + 1) % opt.val_interval == 0:
            best = trainer.val(val_loader, epoch, logger, best)  # 验证


    # ********************** 结束训练 **********************
    print_banner('Training completed')
    # 统计耗时
    total_training_time = time.time() - total_start_time
    total_seconds = int(total_training_time % 60)
    total_minutes = int(total_training_time // 60)
    total_hours = int(total_minutes // 60)
    total_minutes = total_minutes % 60
    print(f"Total training time: {total_hours}h{total_minutes}m{total_seconds}s.")
    logger.write(f"Total training time: {total_hours}h{total_minutes}m{total_seconds}s.\n")
    avg_time_per_epoch = total_training_time / (opt.num_epochs - start_epoch)
    avg_time_per_epoch_minutes = int(avg_time_per_epoch // 60)
    avg_time_per_epoch_seconds = int(avg_time_per_epoch % 60)
    print(f"Average time per epoch: {avg_time_per_epoch_minutes}m{avg_time_per_epoch_seconds}s.")
    logger.write(f"Average time per epoch: {avg_time_per_epoch_minutes}m{avg_time_per_epoch_seconds}s.\n")

    logger.close()


if __name__ == "__main__":
    opt = Opts().parse()
    train(opt)