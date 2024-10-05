import torch
from getpass import getpass
from d2l import torch as d2l
from models.Transformer import TransformerEncoder, TransformerDecoder
from models.decoderOnly import DecoderOnly
from dataset.loadDataWeChat import load_data_seq2seq
from dataset.loadDataDairy import load_data_dairy
from trainingInference.seq2seq import train_seq2seq, predict_seq2seq, bleu
from trainingInference.pretrainSft import pretrain, predict
from evaluation.embedding import visualize_embedding_norm

#a Encoder+Decoder
#|  init
#|1 MHA的线性变换修改为MLP，删除FFN
#b Decoder Only
#|  num_layers=12, lr=0.0005...
#|1 num_layers=32, lr=0.005...
#|2 num_layers=6, lr=0.0005...
#|3 按照b2配置，词向量归一化

#num_hiddens/num_heads必须为整数
num, edition, data_size = 36, 'b3', 10000
num_hiddens, num_layers, dropout = num, 6, 0
batch_size, num_steps, pretrain_num_steps = 64, 16, 128
lrs, nums_epochs, device = [0.0005, 0.0001, 0.00001, 0.000002], [10000, 10000, 55000, 0], d2l.try_gpu(7)
ffn_num_input, ffn_num_hiddens, num_heads = num, num, 12
key_size, query_size, value_size = num, num, num
norm_shape = [num]

# data_iter, vocab, querys, answers = load_data_seq2seq(data_size, batch_size, num_steps)
pretrain_data_iter, pretrain_vocab = load_data_dairy(f'pretrain.tokenizer', batch_size, 128)

# encoder = TransformerEncoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
#                              num_heads, num_layers, dropout)
# decoder = TransformerDecoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
#                              num_heads, num_layers, dropout)
# EncoderDecoderNet = d2l.EncoderDecoder(encoder, decoder)

EvalEmbeddingNet = DecoderOnly(len(pretrain_vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, 12, dropout)
DecoderOnlyNet = DecoderOnly(len(pretrain_vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, num_layers, dropout)


if __name__ == '__main__':
    # train_seq2seq(EncoderDecoderNet, data_iter, lrs, nums_epochs, vocab, [device], pre_train = None)
    pretrain(DecoderOnlyNet, pretrain_data_iter, lrs, nums_epochs, device, edition, pre_train = None)

    # EncoderDecoderNet.load_state_dict(torch.load(f'MyTransformers-a.params'))
    # EncoderDecoderNet = EncoderDecoderNet.to(device)
    # DecoderOnlyNet.load_state_dict(torch.load(f'MyTransformers-{edition}.params'))
    # DecoderOnlyNet.to(device)
    #
    # history = ''
    # while True:
    #     query = getpass('You: ')
    #     # reply, dec_attention_weight_seq = predict_seq2seq(EncoderDecoderNet, query, vocab, vocab, num_steps, [device], True)
    #     reply, history, _ = predict(DecoderOnlyNet, query, history, pretrain_vocab, pretrain_num_steps, device)
    #     print(f'Nyte: {reply}\n')

    # for query, answer in zip(querys[:100], answers[:100]):
    #     reply, dec_attention_weight_seq = predict_seq2seq(EncoderDecoderNet, query, vocab, vocab, num_steps, devices, True)
    #     print(f'{query} => {reply} <=> {answer}, bleu {bleu(reply, answer, k=2) : .3f}')
