import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict

class CropFixAreaMyData(object):
    """针对自定义数据格式的固定区域裁剪增强
    Arguments:
        opt: 配置参数对象
    """
    def __init__(self, opt):
        self.opt = opt
        self.sample_area = (512, 512)  # 固定裁剪区域大小

    def gen_mask(self, targets, rect, frame_id):
        """生成当前帧中在裁剪区域内的目标掩码
        Args:
            targets: 当前帧的目标列表，格式为[frame_id, obj_id, x, y, w, h, cls]
            rect: 裁剪区域 [x1, y1, x2, y2]
            frame_id: 当前帧ID
        Returns:
            mask: 布尔掩码，标识哪些目标在裁剪区域内
        """
        if not targets:
            return np.array([], dtype=bool)
        
        # 提取当前帧的目标
        frame_targets = [t for t in targets if t[0] == frame_id]
        if not frame_targets:
            return np.array([], dtype=bool)
        
        # 计算每个目标的边界框中心点
        centers = []
        for target in frame_targets:
            x, y, w, h = target[2:6]  # x, y, w, h
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # 检查目标中心是否在裁剪区域内
        m1 = (rect[0] <= centers[:, 0]) * (rect[1] <= centers[:, 1])
        m2 = (rect[2] >= centers[:, 0]) * (rect[3] >= centers[:, 1])
        mask = m1 * m2
        
        return mask

    def crop_target(self, target, rect):
        """裁剪目标边界框到指定区域
        Args:
            target: 目标信息 [frame_id, obj_id, x, y, w, h, cls]
            rect: 裁剪区域 [x1, y1, x2, y2]
        Returns:
            cropped_target: 裁剪后的目标信息
        """
        frame_id, obj_id, x, y, w, h, cls = target
        
        # 计算目标边界框的四个角点
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # 与裁剪区域求交集
        new_x1 = max(x1, rect[0])
        new_y1 = max(y1, rect[1])
        new_x2 = min(x2, rect[2])
        new_y2 = min(y2, rect[3])
        
        # 检查是否有有效交集
        if new_x1 >= new_x2 or new_y1 >= new_y2:
            return None
        
        # 转换为相对于裁剪区域的坐标
        new_x = new_x1 - rect[0]
        new_y = new_y1 - rect[1]
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        
        return [frame_id, obj_id, new_x, new_y, new_w, new_h, cls]

    def __call__(self, images, targets):
        """执行数据增强
        Args:
            images: 输入图像序列，形状为(seq_len, 3, h, w)
            targets: 目标字典，key为0,1,...,seq_len-1，value为目标列表
        Returns:
            cropped_images: 裁剪后的图像序列
            cropped_targets: 裁剪后的目标字典
        """
        seq_len, channels, height, width = images.shape
        
        # 检查图像尺寸是否小于裁剪区域，如果是则进行填充
        if height < self.sample_area[1] or width < self.sample_area[0]:
            # 创建填充后的图像
            padded_height = max(height, self.sample_area[1])
            padded_width = max(width, self.sample_area[0])
            images_pad = torch.zeros(seq_len, channels, padded_height, padded_width)
            
            # 填充图像
            for i in range(seq_len):
                # 将图像转换为numpy格式进行填充
                img_np = images[i].permute(1, 2, 0).numpy()
                padded_img = cv2.copyMakeBorder(
                    img_np, 
                    0, max(0, padded_height - height), 
                    0, max(0, padded_width - width), 
                    cv2.BORDER_CONSTANT, 
                    value=(0, 0, 0)
                )
                images_pad[i] = torch.from_numpy(padded_img).permute(2, 0, 1)
            
            height = padded_height
            width = padded_width
        else:
            images_pad = images
        
        # 尝试多次裁剪直到找到包含目标的有效区域
        flag = 'have_box'
        for count in range(37):
            current_images = images_pad.clone()
            
            w = self.sample_area[0]
            h = self.sample_area[1]
            
            # 确定裁剪位置
            if flag == 'have_box':
                # 随机裁剪
                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
            else:
                # 网格采样
                left = (width - w) / 5 * ((count - 1) % 6)
                top = (height - h) / 5 * int((count - 1) / 6)
                if (count - 1) % 6 == 5:
                    left = width - w
                if int((count - 1) / 6) == 5:
                    top = height - h
            
            # 裁剪区域坐标
            rect = np.array([int(left), int(top), int(left + w), int(top + h)])
            
            # 裁剪图像序列
            current_images = current_images[:, :, rect[1]:rect[3], rect[0]:rect[2]]
            
            # 检查是否有目标
            has_targets = any(len(targets.get(i, [])) > 0 for i in range(seq_len))
            if not has_targets:
                print('Attention!!! Images have no targets!!!')
                return current_images, targets
            
            # 检查每一帧是否有目标在裁剪区域内
            valid_crop = True
            for frame_id in range(seq_len):
                frame_targets = targets.get(frame_id, [])
                if frame_targets:
                    mask = self.gen_mask(frame_targets, rect, frame_id)
                    if not mask.any():
                        valid_crop = False
                        break
            
            # 如果没有有效目标且未达到最大尝试次数，继续尝试
            if not valid_crop and count < 36:
                flag = 'no_box'
                continue
            elif not valid_crop and count == 36:
                raise ValueError('Strange thing!!! Images have no targets!!!')
            
            # 裁剪目标
            cropped_targets = {}
            for frame_id in range(seq_len):
                frame_targets = targets.get(frame_id, [])
                if frame_targets:
                    mask = self.gen_mask(frame_targets, rect, frame_id)
                    frame_targets_array = np.array(frame_targets)
                    valid_targets = frame_targets_array[mask]
                    
                    # 裁剪每个有效目标
                    cropped_frame_targets = []
                    for target in valid_targets:
                        cropped_target = self.crop_target(target, rect)
                        if cropped_target is not None:
                            cropped_frame_targets.append(cropped_target)
                    
                    cropped_targets[frame_id] = cropped_frame_targets
                else:
                    cropped_targets[frame_id] = []
            
            return current_images, cropped_targets
        
        # 如果所有尝试都失败，返回原始数据
        return images, targets


class RandomMirrorMyData(object):
    """针对自定义数据格式的随机镜像增强"""
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, images, targets):
        """执行随机镜像增强
        Args:
            images: 输入图像序列，形状为(seq_len, 3, h, w)
            targets: 目标字典
        Returns:
            mirrored_images: 镜像后的图像序列
            mirrored_targets: 镜像后的目标字典
        """
        seq_len, channels, height, width = images.shape
        
        # 检查是否有目标
        has_targets = any(len(targets.get(i, [])) > 0 for i in range(seq_len))
        if not has_targets:
            return images, targets
        
        # 水平镜像
        if random.randint(0, 1):
            images = images.flip(dims=[3])  # 沿宽度维度翻转
            
            # 更新目标坐标
            mirrored_targets = {}
            for frame_id in range(seq_len):
                frame_targets = targets.get(frame_id, [])
                mirrored_frame_targets = []
                
                for target in frame_targets:
                    frame_id, obj_id, x, y, w, h, cls = target
                    # 水平镜像：x坐标变为width - x - w
                    new_x = width - x - w
                    mirrored_frame_targets.append([frame_id, obj_id, new_x, y, w, h, cls])
                
                mirrored_targets[frame_id] = mirrored_frame_targets
            
            targets = mirrored_targets
        
        # 垂直镜像
        if random.randint(0, 1):
            images = images.flip(dims=[2])  # 沿高度维度翻转
            
            # 更新目标坐标
            mirrored_targets = {}
            for frame_id in range(seq_len):
                frame_targets = targets.get(frame_id, [])
                mirrored_frame_targets = []
                
                for target in frame_targets:
                    frame_id, obj_id, x, y, w, h, cls = target
                    # 垂直镜像：y坐标变为height - y - h
                    new_y = height - y - h
                    mirrored_frame_targets.append([frame_id, obj_id, x, new_y, w, h, cls])
                
                mirrored_targets[frame_id] = mirrored_frame_targets
            
            targets = mirrored_targets
        
        return images, targets


class ConvertFromIntsMyData(object):
    """将图像数据类型从整数转换为浮点数
    作用：确保图像数据为float32类型，便于后续处理
    """
    def __call__(self, images, targets):
        """转换图像数据类型
        Args:
            images: 输入图像序列，形状为(seq_len, 3, h, w)
            targets: 目标字典
        Returns:
            converted_images: 转换后的图像序列
            targets: 目标字典（不变）
        """
        # 将图像转换为float32类型
        if images.dtype != torch.float32:
            images = images.float()
        
        return images, targets


class ComposeMyData(object):
    """组合多个数据增强操作"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, targets):
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


class AugmentationMyData(object):
    """完整的数据增强流程"""
    def __init__(self, opt):
        self.opt = opt
        self.augment = ComposeMyData([
            CropFixAreaMyData(opt),
            RandomMirrorMyData(opt),
            ConvertFromIntsMyData(),  # 添加数据类型转换
        ])

    def __call__(self, images, targets):
        """执行完整的数据增强流程
        Args:
            images: 输入图像序列，形状为(seq_len, 3, h, w)
            targets: 目标字典
        Returns:
            augmented_images: 增强后的图像序列
            augmented_targets: 增强后的目标字典
        """
        return self.augment(images, targets)


if __name__ == "__main__":
    """测试数据增强功能"""
    import matplotlib.pyplot as plt
    
    # 创建配置对象
    class Opt:
        def __init__(self):
            self.seq_len = 3  # 序列长度
    
    opt = Opt()
    
    # 构造测试数据
    seq_len = 3
    height, width = 600, 800
    channels = 3
    
    # 创建图像序列 (seq_len, 3, h, w)
    images = torch.randn(seq_len, channels, height, width)
    
    # 创建目标字典 - 演示不同帧有不同数量的目标
    targets = {
        0: [  # 第0帧有3个目标
            [0, 1, 100, 150, 80, 60, 1],   # [frame_id, obj_id, x, y, w, h, cls]
            [0, 2, 300, 200, 100, 80, 2],
            [0, 3, 500, 300, 120, 90, 1]
        ],
        1: [  # 第1帧只有1个目标
            [1, 1, 120, 160, 85, 65, 1],   # 稍微移动的目标
        ],
        2: [  # 第2帧有2个目标
            [2, 1, 140, 170, 90, 70, 1],   # 继续移动的目标
            [2, 2, 340, 220, 110, 90, 2],
        ]
    }
    
    print("原始数据信息:")
    print(f"图像形状: {images.shape}")
    print(f"目标数量: {sum(len(targets[i]) for i in range(seq_len))}")
    for i in range(seq_len):
        print(f"第{i}帧目标: {len(targets[i])}个")
    
    # 创建数据增强器
    augmentor = AugmentationMyData(opt)
    
    # 执行数据增强
    print("\n执行数据增强...")
    augmented_images, augmented_targets = augmentor(images, targets)
    
    print("\n增强后数据信息:")
    print(f"图像形状: {augmented_images.shape}")
    print(f"目标数量: {sum(len(augmented_targets[i]) for i in range(seq_len))}")
    for i in range(seq_len):
        print(f"第{i}帧目标: {len(augmented_targets[i])}个")
        if augmented_targets[i]:
            print(f"  目标详情: {augmented_targets[i]}")
    
    # 测试裁剪功能
    print("\n测试单独裁剪功能:")
    crop_augmentor = CropFixAreaMyData(opt)
    cropped_images, cropped_targets = crop_augmentor(images, targets)
    print(f"裁剪后图像形状: {cropped_images.shape}")
    print(f"裁剪后目标数量: {sum(len(cropped_targets[i]) for i in range(seq_len))}")
    
    # 测试镜像功能
    print("\n测试单独镜像功能:")
    mirror_augmentor = RandomMirrorMyData(opt)
    mirrored_images, mirrored_targets = mirror_augmentor(images, targets)
    print(f"镜像后图像形状: {mirrored_images.shape}")
    print(f"镜像后目标数量: {sum(len(mirrored_targets[i]) for i in range(seq_len))}")
    
    # 测试数据类型转换功能
    print("\n测试数据类型转换功能:")
    convert_augmentor = ConvertFromIntsMyData()
    # 创建整数类型的图像进行测试
    int_images = torch.randint(0, 255, (seq_len, channels, height, width), dtype=torch.uint8)
    converted_images, converted_targets = convert_augmentor(int_images, targets)
    print(f"原始图像数据类型: {int_images.dtype}")
    print(f"转换后图像数据类型: {converted_images.dtype}")
    print(f"转换后图像形状: {converted_images.shape}")
    
    # # 可视化测试（可选）
    # try:
    #     # 显示原始图像的第一帧
    #     plt.figure(figsize=(15, 5))
        
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(images[0].permute(1, 2, 0).numpy())
    #     plt.title("原始图像 (第0帧)")
    #     plt.axis('off')
        
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(augmented_images[0].permute(1, 2, 0).numpy())
    #     plt.title("增强后图像 (第0帧)")
    #     plt.axis('off')
        
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(cropped_images[0].permute(1, 2, 0).numpy())
    #     plt.title("裁剪后图像 (第0帧)")
    #     plt.axis('off')
        
    #     plt.tight_layout()
    #     plt.show()
        
    #     print("\n可视化测试完成!")
        
    # except Exception as e:
    #     print(f"可视化测试失败: {e}")
    #     print("这可能是因为没有显示设备或matplotlib配置问题")
    
    print("\n测试完成!")
    
    # 额外测试：展示代码如何处理不同长度的目标列表
    print("\n" + "="*50)
    print("额外测试：不同长度目标列表的处理")
    print("="*50)
    
    # 创建更复杂的测试数据
    complex_targets = {
        0: [  # 第0帧：5个目标
            [0, 1, 50, 50, 60, 40, 1],
            [0, 2, 150, 100, 80, 60, 2],
            [0, 3, 250, 150, 70, 50, 1],
            [0, 4, 350, 200, 90, 70, 2],
            [0, 5, 450, 250, 100, 80, 1]
        ],
        1: [],  # 第1帧：无目标
        2: [  # 第2帧：1个目标
            [2, 1, 200, 150, 100, 80, 1]
        ],
        3: [  # 第3帧：3个目标
            [3, 1, 100, 100, 80, 60, 1],
            [3, 2, 300, 200, 90, 70, 2],
            [3, 3, 500, 300, 110, 90, 1]
        ],
        4: [  # 第4帧：2个目标
            [4, 1, 150, 120, 85, 65, 1],
            [4, 2, 400, 250, 95, 75, 2]
        ]
    }
    
    # 创建对应的图像序列
    complex_images = torch.randn(5, channels, height, width)
    
    print("复杂测试数据信息:")
    print(f"图像形状: {complex_images.shape}")
    for i in range(5):
        print(f"第{i}帧目标数量: {len(complex_targets.get(i, []))}")
    
    # 测试复杂数据
    complex_augmented_images, complex_augmented_targets = augmentor(complex_images, complex_targets)
    
    print("\n复杂数据增强后:")
    print(f"图像形状: {complex_augmented_images.shape}")
    for i in range(5):
        target_count = len(complex_augmented_targets.get(i, []))
        print(f"第{i}帧目标数量: {target_count}")
        if target_count > 0:
            print(f"  目标详情: {complex_augmented_targets[i]}")
    
    print("\n代码成功处理了不同长度的目标列表!") 