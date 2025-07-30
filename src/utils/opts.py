import argparse
import os
from datetime import datetime
from src.utils.print_utils import print_banner

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="RSIPAC MP2Net Training Options")

        # basic options
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')
        self.parser.add_argument('--num_classes', type=int, default=1,
                                 help='4 classes for car, airplane, ship and train.')
        self.parser.add_argument('--max_objs', type=int, default=350,
                                 help='max number of output objects.')
        
        # system options
        self.parser.add_argument('--gpus', default='0, 1',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')
        self.parser.add_argument('--model_path', type=str, default='',
                                 help='path to model')

        # training options
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='14',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=20,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=2,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=4,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seq_len', type=int, default=5,
                                 help='number of images for per sample. Currently supports 5.')
        self.parser.add_argument('--amp', action='store_true',
                                 help='enable automatic mixed precision (AMP) training')

        # dataset options
        self.parser.add_argument('--data_dir', type=str, default='./data/dataset',
                                 help='path to dataset directory')

        # saving options
        self.parser.add_argument('--save_dir', type=str, default='./experiments',
                                 help='path to save outputs')

    def parse(self):
        opt = self.parser.parse_args()

        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(step) for step in opt.lr_step.split(',')]

        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        opt.save_dir = os.path.join(opt.save_dir, now)

        print_banner("Print options")
        print(opt)

        return opt