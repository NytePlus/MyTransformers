import os
import re
import collections
import torch
from torch.utils import data

class Vocabulary():
    def __init__(self, tokens=None, reserved_tokens=None, min_freq=0):
        if not tokens:
            tokens = []
        if not reserved_tokens:
            reserved_tokens = []
        counter = count_corpus(tokens)
        tokens_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in tokens_freq
                        if token not in uniq_tokens and freq >= min_freq]
        self.id_to_token, self.token_to_id = [], dict()
        for token in uniq_tokens:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def __getitem__(self, token):
        if not isinstance(token, (tuple, list)):
            return self.token_to_id.get(token, self.unk)
        return [self.__getitem__(i) for i in token]

    def __len__(self):
        return len(self.id_to_token)

    def to_token(self, index):
        if not isinstance(index, (tuple, list)):
            return self.id_to_token[index]
        return [self.to_token(i) for i in index]


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], (tuple, list)):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(text):
    tokens, emoji_pos, pre_i = [], [], 0
    for m in re.finditer('\[(.*?)\]', text):
        emoji_pos.append([m.start(), m.end()])
    for area in emoji_pos:
        for i in range(pre_i, area[0]):
            tokens.append(text[i])
        # print(list(range(* area)))
        tokens.append(text[area[0]: area[1]])
        pre_i = area[1]
    return tokens + [token for token in text[pre_i:]]


def txt_to_text(data_dir):
    def is_valid(tokens):
        return tokens not in ['[图片]\n', '[表情包]\n'] and len(tokens) < 30

    people_list = os.listdir(data_dir)
    conversations, querys, answers = [], [], []
    for people in people_list:
        if os.path.splitext(people)[1] == '':
            file = os.path.join(data_dir, people, people + '.txt')
            data, query, answer = [], None, None
            with open(file, 'r', encoding='utf-8') as f:
                raw_data = f.readlines()
            for line in raw_data:
                if line == '\n' or line[:3] == '引用:':
                    continue
                if re.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d', line) is not None:
                    data.append(line)
                elif re.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d', data[-1]) is not None:
                    data.append(line)
                else:
                    data[-1] += line
            for date_time_name, tokens in zip(data[0:: 2], data[1:: 2]):
                if not is_valid(tokens):
                    continue
                if date_time_name[- 3:] != ':D\n':
                    query = tokens[: - 1]
                elif date_time_name[- 3:] == ':D\n' and query is not None:
                    answer = tokens[: - 1]
                    conversations.append([query, answer])
                    querys.append(query)
                    answers.append(answer)
                    query, answer = None, None
    return conversations, querys, answers

def truncate_pad(line, padding_token, num_steps):
    if len(line) > num_steps:
        return line[: num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))

def build_array(lines, vocab, num_steps):
    lines = [vocab[l] + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, vocab['<pad>'], num_steps) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_weChat(data_size, batch_size, num_steps):
    conversations, querys, answers = txt_to_text('/home/wcc/data/weChat/聊天记录')
    source = [tokenize(query) for query in querys[: data_size]]
    target = [tokenize(answer) for answer in answers[: data_size]]
    vocab = Vocabulary(tokens=source + target, reserved_tokens=['<pad>', '<eos>', '<bos>'], min_freq = 0)
    source_array, src_valid_len = build_array(source, vocab, num_steps)
    target_array, tgt_valid_len = build_array(target, vocab, num_steps)
    dataset = data.TensorDataset(source_array, src_valid_len, target_array, tgt_valid_len)
    data_iter = data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

    print(f'conversation_size: {len(conversations)}')
    print(f'vocab_size: {len(vocab)}')
    return data_iter, vocab, querys[: data_size], answers[: data_size]


