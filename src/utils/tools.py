import datetime

def print_banner(msg, width=80, fill_char='='):
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 拼接时间戳和消息
    full_msg = f' {timestamp} | {msg} '
    # 绿色ANSI转义码
    GREEN = '\033[92m'
    RESET = '\033[0m'
    # 居中填充
    banner = full_msg.center(width, fill_char)
    # 打印绿色banner
    print(f"\n{GREEN}{banner}{RESET}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count