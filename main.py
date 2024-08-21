import collections
import torch
import math
from torch import nn
from getpass import getpass
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l
from Transformer import TransformerEncoder, TransformerDecoder
from load_data_weChat import load_data_weChat, tokenize, truncate_pad

#num_hiddens/num_heads必须为整数
num, edition, data_size = 36, 'a', 10000
num_hiddens, num_layers, dropout = num, 12, 0
batch_size, num_steps = 64, 16
lrs, nums_epochs, devices = [0.0001, 0.0001, 0.00001, 0.000001], [500, 0, 0, 0], [d2l.try_gpu(6)]#, d2l.try_gpu(1), d2l.try_gpu(2)]
ffn_num_input, ffn_num_hiddens, num_heads = num, num, 12
key_size, query_size, value_size = num, num, num
norm_shape = [num]

data_iter, vocab, querys, answers = load_data_weChat(data_size, batch_size, num_steps)


encoder = TransformerEncoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)


def train_seq2seq(net, data_iter, lrs, nums_epochs, tgt_vocab, devices, log_dir, pre_train = None):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:  # ?
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    if pre_train == None:
        net.apply(xavier_init_weights)
    else:
        net.load_state_dict(torch.load(pre_train))
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss = d2l.MaskedSoftmaxCELoss()
    writer = SummaryWriter(log_dir = log_dir)
    net.train()
    for i, (lr, num_epochs) in enumerate(zip(lrs, nums_epochs)):
        updater = torch.optim.Adam(net.parameters(), lr = lr)
        finished_epochs = sum(nums_epochs[:i])
        for epoch in range(num_epochs):
            timer = d2l.Timer()
            metric = d2l.Accumulator(2)
            for batch in data_iter:
                X, X_valid_len, Y, Y_valid_len = [x.to(devices[0]) for x in batch]
                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device = devices[0]).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, : -1]], 1)
                # 输出要加入<bos>但是输入不需要
                Y_hat = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)

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
                writer.add_scalar('Loss', metric[0] / metric[1], epoch + finished_epochs)
            if (epoch + 1) % 100 == 0:
                torch.save(net.module.state_dict(), f'MyTransformers-{edition}.params')
                print(f'epoch {finished_epochs + epoch + 1} finished.')
    writer.close()
    print(f'loss {metric[0] / metric[1] : .3f}, {metric[1] / timer.stop() : .1f} tokens/sec on {str(devices)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, devices, save_attention_weights = False):
    net.eval()
    src_tokens = src_vocab[tokenize(src_sentence)] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device = devices[0])
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # 添加批量轴，转换成long类型，转移到cuda上
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device = devices[0]), dim = 0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype = torch.long, device = devices[0]), dim = 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim = 2)
        pred = dec_X.squeeze(dim = 0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ''.join(tgt_vocab.to_token(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = [token for token in pred_seq], [token for token in label_seq]
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        if len_pred - n + 1 == 0:
            score = 0
        else:
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


if __name__ == '__main__':
    #for query, answer in zip(querys[:10], answers[:10]):
    #    print(f'{query} => {answer}')
    train_seq2seq(net, data_iter, lrs, nums_epochs, vocab, devices, log_dir = f'/home/wcc/MyTransormers-{edition}/', pre_train = None)

    # net.load_state_dict(torch.load(f'seq2seq_zhihu{edition}.params'))
    # net = net.to(devices[0])
    #
    # '''
    # while True:
    #     query = getpass('You: ')
    #     reply, dec_attention_weight_seq = predict_seq2seq(net, query, vocab, vocab, num_steps, devices, True)
    #     print(f'Nyte: {reply}\n')
    # '''
    #
    # for query, answer in zip(querys[:20], answers[:20]):
    #     reply, dec_attention_weight_seq = predict_seq2seq(net, query, vocab, vocab, num_steps, devices, True)
    #     print(f'{query} => {reply} <=> {answer}, bleu {bleu(reply, answer, k=2) : .3f}')