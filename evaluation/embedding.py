import torch
from torch.utils.tensorboard import SummaryWriter

'''
每个词向量的范数，处在4到14之间
'''

def visualize_embedding_norm(model, ckpt, log_dir = f'/home/wcc/logs/MyTransformers'):
    writer = SummaryWriter(log_dir = log_dir)
    model.load_state_dict(torch.load(ckpt))
    for name, param in model.named_parameters():
        if 'embed' in name:
            embedding_weights = param
            break
    embedding_norms = torch.norm(embedding_weights, p = 2, dim = 1)

    sorted_indices = torch.argsort(embedding_norms)
    for id, idx in enumerate(sorted_indices):
        writer.add_scalar(f'Embedding_L2_Norm', embedding_norms[idx].item(), id)

    writer.close()
