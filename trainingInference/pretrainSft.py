import math
import torch
from torch import nn
from d2l import torch as d2l
from dataset.loadDataWeChat import tokenize, truncate_pad
from torch.utils.tensorboard import SummaryWriter

def evaluate_accuracy(Y_hat, Y, valid_len):
    weights = torch.ones_like(Y)
    mask = torch.arange((Y.shape[1]), dtype = torch.float32, device = Y.device)[None, :] < valid_len[:, None]
    weights[~ mask] = 0
    return ((Y_hat.argmax(dim = -1) == Y).int() * weights).sum()

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred [batch_size, num_steps, vocab_size]
    # label [batch_size, num_steps]
    # valid_len [batch_size, num_masks, 2]
    def forward(self, pred, label, valid_len):
        weights = self.sequence_mask(torch.ones_like(label), valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).sum(dim = 1)
        return weighted_loss

    def sequence_mask(self, array, masks):
        if len(masks.shape) == 1:
            maxlen = array.shape[1]
            mask = torch.arange((maxlen), dtype = torch.float32,
                                device = array.device)[None, :] < masks[:, None]
            array[~mask] = 0
        else :
            for i in range(masks.shape[0]):
                for j in range(len(masks[i])):
                    array[i][masks[i][j][0] : masks[i][j][1]] = 0
        return array

def pretrain(net, data_iter, lrs, nums_epochs, device, edition, log_dir = f'/home/wcc/logs/MyTransormers', pre_train = None):
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
    loss = MaskedSoftmaxCELoss()
    writer = SummaryWriter(log_dir = log_dir)
    net.to(device)
    net.train()
    for i, (lr, num_epochs) in enumerate(zip(lrs, nums_epochs)):
        updater = torch.optim.Adam(net.parameters(), lr = lr)
        finished_epochs = sum(nums_epochs[:i])
        for epoch in range(num_epochs):
            timer = d2l.Timer()
            metric = d2l.Accumulator(3)
            for batch in data_iter:
                X, Y, valid_lens = batch[0][:, : -1].to(device), batch[0][:, 1 :].to(device), batch[1].to(device)
                Y_hat, _ = net(X, net.init_state())
                # from main import pretrain_vocab
                # print(f'Y[0][0]: {"".join([pretrain_vocab.to_token(t) for t in Y[0]])} Y_hat[0][0]: {"".join([pretrain_vocab.to_token(t) for t in Y_hat[0].argmax(dim = 1)])} Y_hat[0].max: {Y_hat[0].max()} Y_hat[0][0][Y[0][0]]: {Y_hat[0][0][Y[0][0]]}')
                l = loss(Y_hat, Y, valid_lens)
                acc = evaluate_accuracy(Y_hat, Y, valid_lens)
                updater.zero_grad()

                l.sum().backward()
                for i in l:
                    assert not math.isnan(i), 'loss apear not a number(nan).'
                updater.step()

                num_tokens = valid_lens.sum()
                with torch.no_grad():
                    metric.add(l.sum(), acc, num_tokens)
            print(metric[0] / metric[2], metric[1] / metric[2])
            if (epoch + 1) % 10 == 0:
                writer.add_scalar(f'MyTransformers-{edition} Loss', metric[0] / metric[2], epoch + finished_epochs)
                writer.add_scalar(f'MyTransformers-{edition} Accuracy', metric[1] / metric[2], epoch + finished_epochs)
            if (epoch + 1) % 100 == 0:
                from main import edition
                torch.save(net.state_dict(), f'MyTransformers-{edition}.params')
                print(f'epoch {finished_epochs + epoch + 1} finished.')
    writer.close()
    print(f'loss {metric[0] / metric[1] : .3f}, {metric[1] / timer.stop() : .1f} tokens/sec on {str(device)}')

def predict(net, input, history, vocab, num_steps, device, save_attention_weights = False):
    net.train()
    tokens = vocab[tokenize(history + input)]

    X = torch.unsqueeze(torch.tensor(tokens, dtype = torch.long, device = device), dim = 0)
    X, state = net(X, net.init_state())
    print(''.join([vocab.to_token(t) for t in X[0].argmax(dim = 1)]))
    X = X[:, -1 :].argmax(dim = 2)

    net.eval()
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        print(f'out: {vocab.to_token(X.squeeze(dim = 0).type(torch.int32).item())}')
        Y, state = net(X, state)
        X = Y.argmax(dim = 2)
        pred = X.squeeze(dim = 0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == vocab['<eos>'] or len(history) + len(output_seq) > num_steps:
            break
        output_seq.append(pred)
    output = ''.join(vocab.to_token(output_seq))
    return output, history + output, attention_weight_seq