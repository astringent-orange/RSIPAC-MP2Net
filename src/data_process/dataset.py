import cv2
import os
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset
from src.utils.data_augmentation import Augmentation 
from src.data_prcoess.image import (
    draw_umich_gaussian, gaussian_radius, flip, color_aug, 
    get_affine_transform, affine_transform, draw_msra_gaussian,
    draw_dense_reg
)

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

        # 数据集长度
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


        # ############## 读取图片文件 ###############
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
        
        

        # ################ 读取标签文件 ###############
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
                targets[self.seq_len - 1].append(line[0:7])  # 将当前帧的目标信息添加到targets中
        
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
                            targets[self.seq_len - i - 2].append(line[0:7])


        # ################ 数据处理 ###############
        height, width = images.shape[0] - images.shape[0] % 32, images.shape[1] - images.shape[1] % 32
        output_height, output_width = height // self.down_ratio, width // self.down_ratio

        if self.mode == 'train':
            images, targets = self.augment(images, targets)                             # 数据增强
            targets = self._generate_targets(targets, output_height, output_width)      # 生成目标特征图

        images = np.transpose(images, (3, 2, 0, 1))   # Transpose to (H, W, 3, N) -> (N, 3, H, W)以便pytorch处理
        images = images[:, :, 0:height, 0:width]      # 裁剪图片尺寸为32的倍数，以便DLA Backbone处理

        return images, targets


    # ################# 特征图生成函数 ###############
    def _get_transoutput(self, c, s, height, width):
        """计算仿射变换矩阵，用于标签映射"""
        from src.utils.image import get_affine_transform
        trans_output = get_affine_transform(c, s, 0, [width, height])
        return trans_output

    def transform_box(self, bbox, transform, output_w, output_h):
        """对bbox做仿射变换，得到中心点和高斯半径，用于生成heatmap"""
        from src.utils.image import affine_transform, gaussian_radius
        bbox[:2] = affine_transform(bbox[:2], transform)
        bbox[2:] = affine_transform(bbox[2:], transform)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h = np.clip(h, 0, output_h - 1)
        w = np.clip(w, 0, output_w - 1)
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct[0] = np.clip(ct[0], 0, output_w - 1)
        ct[1] = np.clip(ct[1], 0, output_h - 1)
        return bbox, ct, radius

    def _get_feature(self, targets, height, width):
        """将targets转换为热力图等特征格式
        Args:
            targets: 目标字典，格式为{frame_id: [[frame_id, obj_id, x, y, w, h, cls], ...]}
            height: 图像高度
            width: 图像宽度
        Returns:
            ret: 包含各种特征图的字典
        """
        # 计算中心点和缩放因子
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(width, height) * 1.0
        
        # 计算输出特征图尺寸
        output_h = height // self.down_ratio
        output_w = width // self.down_ratio
        trans_output = self._get_transoutput(c, s, output_h, output_w)

        # 初始化各种特征图
        hm = np.zeros((self.seq_len, self.num_classes, output_h, output_w), dtype=np.float32)       # 目标类别热力图 (N, cls_num, H, W)
        hm_seq = np.zeros((self.seq_len, 1, output_h, output_w), dtype=np.float32)                  # 目标存在性热力图 (N, 1, H, W)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)                                         # 目标宽高 (max_objs, 2)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)                                        # 目标中心点偏移 (max_objs, 2)
        ind = np.zeros((self.max_objs), dtype=np.int64)                                             # 目标中心点在特征图上的索引 (max_objs)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                                        # 目标mask, 有效目标为1 (max_objs)
        
        # 时序相关特征
        ind_dis = np.zeros((self.seq_len - 1, self.max_objs), dtype=np.int64)                       # 前序帧目标中心点索引 (N-1, max_objs)
        dis_mask = np.zeros((self.seq_len - 1, self.max_objs), dtype=np.uint8)                      # 前序帧目标mask (N-1, max_objs)
        dis = np.zeros((self.seq_len - 1, self.max_objs, 2), dtype=np.float32)                      # 前序帧目标中心点位移 (N-1, max_objs, 2)
        
        gt_det = []
        obj_count = 0  # 当前帧目标计数
        
        # 处理当前帧（最后一帧）
        current_frame_id = self.seq_len - 1
        current_targets = targets.get(current_frame_id, [])
        
        for target in current_targets:
            if obj_count >= self.max_objs:
                break
                
            frame_id, obj_id, x, y, w, h, cls = target
            
            # 转换为bbox格式 [x1, y1, x2, y2]
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            
            # 仿射变换bbox到输出空间
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)
            
            if h > 0 and w > 0:
                # 计算高斯半径
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                
                # 计算中心点
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct[0] = np.clip(ct[0], 0, output_w - 1)
                ct[1] = np.clip(ct[1], 0, output_h - 1)
                ct_int = ct.astype(np.int32)
                
                # 对于单类别，所有目标都映射到类别0
                if self.num_classes == 1:
                    cls_id = 0  # 所有目标都映射到类别0
                else:
                    cls_id = int(cls) - 1  # 原始逻辑
                
                # 绘制类别热力图
                draw_umich_gaussian(hm[current_frame_id][cls_id], ct_int, radius)
                # 绘制存在性热力图
                draw_umich_gaussian(hm_seq[current_frame_id][0], ct_int, radius)
                
                # 记录目标信息
                wh[obj_count] = [w, h]
                ind[obj_count] = ct_int[1] * output_w + ct_int[0]
                reg[obj_count] = ct - ct_int
                reg_mask[obj_count] = 1
                
                # 添加到gt_det
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                              ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
                
                obj_count += 1
        
        # 处理前序帧的时序信息
        for i in range(self.seq_len - 1):
            frame_id = i
            frame_targets = targets.get(frame_id, [])
            
            for target in frame_targets:
                frame_id, obj_id, x, y, w, h, cls = target
                
                # 转换为bbox格式 [x1, y1, x2, y2]
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                
                # 仿射变换bbox到输出空间
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                h = np.clip(h, 0, output_h - 1)
                w = np.clip(w, 0, output_w - 1)
                
                if h > 0 and w > 0:
                    # 计算高斯半径
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    
                    # 计算中心点
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct[0] = np.clip(ct[0], 0, output_w - 1)
                    ct[1] = np.clip(ct[1], 0, output_h - 1)
                    ct_int = ct.astype(np.int32)
                    
                    # 对于单类别，所有目标都映射到类别0
                    if self.num_classes == 1:
                        cls_id = 0  # 所有目标都映射到类别0
                    else:
                        cls_id = int(cls) - 1  # 原始逻辑
                    
                    # 绘制类别热力图
                    draw_umich_gaussian(hm[frame_id][cls_id], ct_int, radius)
                    # 绘制存在性热力图
                    draw_umich_gaussian(hm_seq[frame_id][0], ct_int, radius)
                    
                    # 记录时序信息
                    if obj_id <= self.max_objs - 1:  # 确保索引不越界
                        ind_dis[frame_id][obj_id] = ct_int[1] * output_w + ct_int[0]
                        dis_mask[frame_id][obj_id] = 1
                        # 计算位移（这里简化，实际需要与下一帧关联）
                        dis[frame_id][obj_id] = [0, 0]  # 简化处理
        
        # 返回特征字典
        ret = {
            'hm': hm,                    # 类别热力图
            'hm_seq': hm_seq,            # 存在性热力图
            'reg_mask': reg_mask,        # 目标mask
            'ind': ind,                  # 目标索引
            'wh': wh,                    # 目标宽高
            'reg': reg,                  # 目标偏移
            'dis_ind': ind_dis,          # 时序目标索引
            'dis': dis,                  # 时序目标位移
            'dis_mask': dis_mask,        # 时序目标mask
            'gt_det': gt_det             # 目标检测结果
        }
        
        return ret