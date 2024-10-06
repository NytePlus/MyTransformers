import numpy as np
from torch.utils.tensorboard import SummaryWriter

'''
如果正确率是0.7，那么期望正确生成的语句长度为7.9
如果正确率是0.8，那么期望正确生成的语句长度为20.0
如果正确率是0.9，那么期望正确生成的语句长度为89.2
'''

def f(p, n):
    return (p - (n + 1) * p ** (n + 1) + n * p ** (n + 2)) / (1 - p) ** 2

def visualize_acc_step(log_dir = f'/home/wcc/logs/MyTransformers'):
    writer = SummaryWriter(log_dir = log_dir)
    x = np.linspace(0, 0.95, 1000)
    for xi in x:
        writer.add_scalar('Acc-Step', f(xi, 128), xi * 1000)

    writer.close()