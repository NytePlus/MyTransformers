import math
import torch
from torch import nn
from d2l import torch as d2l
from dataset.loadDataWeChat import tokenize, truncate_pad
from torch.utils.tensorboard import SummaryWriter

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred [batch_size, num_steps, vocab_size]
    # label [batch_size, num_steps]
    # valid_len [batch_size, num_masks, 2]
    def forward(self, pred, label, valid_len):
        weights = self.sequence_mask(torch.ones_like(label), valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weighted_loss

    def sequence_mask(self, array, masks):
        for i in range(masks.shape[0]):
            for j in range(len(masks[i])):
                array[i][masks[i][j][0] : masks[i][j][1]] = 0
        return array

def pretrain(net, data_iter, lrs, nums_epochs, vocab, devices, log_dir = f'/home/wcc/MyTransormers', pre_train = None):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    if pre_train == None:
        net.apply(xavier_init_weights)
    else:
        net.load_state_dict(torch.load(pre_train))
    net = nn.DataParallel(net, device_ids = devices).to(devices[0])
    loss = nn.CrossEntropyLoss(reduction = 'none')
    writer = SummaryWriter(log_dir = log_dir)
    net.train()
    for i, (lr, num_epochs) in enumerate(zip(lrs, nums_epochs)):
        updater = torch.optim.Adam(net.parameters(), lr = lr)
        finished_epochs = sum(nums_epochs[:i])
        for epoch in range(num_epochs):
            timer = d2l.Timer()
            metric = d2l.Accumulator(2)
            for batch in data_iter:

                bos = torch.tensor([vocab['<bos>']] * Y.shape[0], device = devices[0]).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, : -1]], 1)
                Y_hat = net(X, net.init_state())
                l = loss(Y_hat, Y)

                updater.zero_grad()

                l.sum().backward()
                for i in l:
                    assert not math.isnan(i), 'loss apear not a number(nan).'
                updater.step()

                num_tokens = Y_valid_len.sum()
                with torch.no_grad():
                    metric.add(l.sum(), num_tokens)
            print(metric[0] / metric[1])
            if (epoch + 1) % 10 == 0:
                writer.add_scalar('MyTransformers-a1 Loss', metric[0] / metric[1], epoch + finished_epochs)
            if (epoch + 1) % 100 == 0:
                from main import edition
                torch.save(net.module.state_dict(), f'MyTransformers-{edition}.params')
                print(f'epoch {finished_epochs + epoch + 1} finished.')
    writer.close()
    print(f'loss {metric[0] / metric[1] : .3f}, {metric[1] / timer.stop() : .1f} tokens/sec on {str(devices)}')