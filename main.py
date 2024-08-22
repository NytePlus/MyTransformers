import torch
from d2l import torch as d2l
from models.Transformer import TransformerEncoder, TransformerDecoder
from dataset.loadDataWeChat import load_data_seq2seq
from trainingInference.seq2seq import train_seq2seq, predict_seq2seq

#num_hiddens/num_heads必须为整数
num, edition, data_size = 36, 'a1', 10000
num_hiddens, num_layers, dropout = num, 12, 0
batch_size, num_steps = 64, 16
lrs, nums_epochs, devices = [0.0001, 0.00001, 0.000001], [1500, 500, 0], [d2l.try_gpu(6)]#, d2l.try_gpu(1), d2l.try_gpu(2)]
ffn_num_input, ffn_num_hiddens, num_heads = num, num, 12
key_size, query_size, value_size = num, num, num
norm_shape = [num]

data_iter, vocab, querys, answers = load_data_seq2seq(data_size, batch_size, num_steps)


encoder = TransformerEncoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(vocab), query_size, key_size, value_size, num_hiddens, ffn_num_hiddens, norm_shape,
                             num_heads, num_layers, dropout)
EncoderDecoderNet = d2l.EncoderDecoder(encoder, decoder)


if __name__ == '__main__':
    #for query, answer in zip(querys[:10], answers[:10]):
    #    print(f'{query} => {answer}')
    train_seq2seq(EncoderDecoderNet, data_iter, lrs, nums_epochs, vocab, devices, pre_train = None)

    EncoderDecoderNet.load_state_dict(torch.load(f'MyTransformers-{edition}.params'))
    EncoderDecoderNet = EncoderDecoderNet.to(devices[0])

    '''
    while True:
        query = getpass('You: ')
        reply, dec_attention_weight_seq = predict_seq2seq(net, query, vocab, vocab, num_steps, devices, True)
        print(f'Nyte: {reply}\n')
    '''

    # for query, answer in zip(querys[:20], answers[:20]):
    #     reply, dec_attention_weight_seq = predict_seq2seq(net, query, vocab, vocab, num_steps, devices, True)
    #     print(f'{query} => {reply} <=> {answer}, bleu {bleu(reply, answer, k=2) : .3f}')