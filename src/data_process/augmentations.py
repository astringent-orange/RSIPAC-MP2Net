class Augmentaions():
    def __init__(self, opt):
        self.mode = mode

    def __call__(self, images, targets):
        if self.mode == 'train':
            # 在这里添加训练模式下的图像增强逻辑
            pass
        elif self.mode == 'val':
            # 在这里添加验证模式下的图像处理逻辑
            pass
        return images, targets