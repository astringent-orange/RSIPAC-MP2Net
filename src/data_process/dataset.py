import cv2
import os
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset
from src.utils.data_augmentation import Augmentation 

class RSIPACDatset(Dataset):
    def __init__(self, opt, mode='train'):
        # opt参数
        self.otp = opt
        self.seq_len = otp.seq_len
        self.max_objs = opt.max_objs
        self.num_classes = opt.num_classes
        self.down_ratio = opt.down_ratio

        # 数据集模式
        self.mode = mode
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode must be one of 'train', 'val', or 'test'.")
        
        # 数据路径
        self.data_path = os.path.join(opt.data_dir, mode)
        self.images_dir = os.path.join(self.data_path, 'images')
        self.labels_dir = os.path.join(self.data_path, 'labels')
        self.images_list = sorted(os.listdir(self.images_dir))
        self.labels_list = sorted(os.listdir(self.labels_dir))

        self.num_samples = len(self.images_list)

        # 图片归一化参数
        self.mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

        print(f"Loaded {self.num_samples} {self.mode} samples")

        # 数据增强
        if mode == 'train':
            self.augment = Augmentaions(opt)

    def __len__(self):
        return self.num_samples

    def __getitem__(self. idx):
        image_path = os.path.join(self.images_dir, self.images_list[idx])
        label_path = os.path.join(self.labels_dir, self.labels_list[idx])
        image_name = self.images_list[idx].split('.')[0]    # '1-2_000001'
        video_name = image_name.split('_')[0]               # '1-2'
        frame_id = int(image_name.split('_')[1])            # 1


        ############### 读取图片文件 ###############
        # 构建图像序列 (N, 3, H, W)

        # 先读取当前帧图像
        image_0 = cv2.imread(image_path)
        image_0 = (image_0.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        image_0 = (image_0 - self.mean) / self.std      # Normalize using mean and std
        images = np.zeros([image_0.shape[0], image_0.shape[1], 3, self.seq_len])  # (H, W, 3, N)
        images[:, :, :, self.seq_len - 1] = image_0

        # 再读取前序帧图像
        for i in range(self.seq_len - 1):
            # 倒序填充图像
            idx_i = idx - i - 1
            if idx_i < 0:
                images[:, :, :, self.seq_len - i - 2] = images[:, :, :, self.seq_len - i - 1]   # 索引小于0
                
            else:
                image_path_i = os_path.join(self.imaes_dir, self.image_list[idx_i])   
                video_name_i = self.images_list[idx_i].split('_')[0]    # 该帧的video_name
                if video_name_i != video_name:                          # 如果视频名不一致
                    images[:, :, :, self.seq_len - i - 2] = images[:, :, :, self.seq_len - i - 1]
                else:
                    image_i = cv2.imread(image_path_i)
                    image_i = (image_i.astype(np.float32) / 255.0)  # Normalize to [0, 1]
                    image_i = (image_i - self.mean) / self.std      # Normalize using mean and std
                    images[:, :, :, self.seq_len - i - 2] = image_i
        
        images = np.transpose(images, (3, 2, 0, 1))  # Transpose to (H, W, 3, N) -> (N, 3, H, W)以便pytorch处理
        height, width = images.shape[2] - images.shape[2] % 32, images.shape[3] - images.shape[3] % 32
        images = images[:, :, 0:height, 0:width]      # 裁剪图片尺寸为32的倍数，以便DLA Backbone处理
        

        ################# 读取标签文件 ###############
        # 因为每一帧中的目标数目不一致，所以返回结果无法是矩阵
        # 使用字典类型作为targets返回，key为N帧中的序号，value为每一帧的目标信息列表
        # targets= {0: [[obj_id, x, y, w, h, cls],[],...], 1: [[],[]], ..., N-1: [[],[]]}

        targets = defaultdict(list)    # 初始化标签列表

        # 先读取当前帧标签
        with open(label_path, 'r') as f:
            for id, line in enumerate(f.readlines()):
                if id >= self.max_objs:
                    break
                line = int(line.strip().split(' '))
                targets[self.seq_len - 1].append(line)  # 将当前帧的目标信息添加到targets中
        
        # 再读取前序帧标签
        for i in range(self.seq_len - 1):
            idx_i = idx - i - 1
            if idx_i < 0:
                targets[self.seq_len - i - 2] = targets[self.seq_len - i - 1]
            else:
                label_path_i = os.path.join(self.labels_dir, self.labels_list[idx_i])
                video_name_i = self.labels_list[idx_i].split('_')[0]
                if video_name_i != video_name:  # 如果视频名不一致
                    targets[self.seq_len - i - 2] = targets[self.seq_len - i - 1]
                else:
                    with open(label_path_i, 'r') as f:
                        for id, line in enumerate(f.readlines()):
                            if id >= self.max_objs:
                                break
                            line = int(line.strip().split(' '))
                            targets[self.seq_len - i - 2].append(line)
                



        ################## 训练数据集特殊处理 ###########
        if mode == 'train':
            # 如果是训练模式，对images进行数据增强且targets返回特征图格式
            images, targets = augment(images, targets)

        return images, targets

        def _generate_targets(self, targets):
            # 生成特征图格式的targets
            # 这里需要根据具体任务定义如何将targets转换为特征图格式
            pass