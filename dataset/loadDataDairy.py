import os
import re
import torch
from tqdm import tqdm
from torch.utils import data
from .loadDataWeChat import Vocabulary, tokenize, build_array

num_steps = 128

def preprocess(data_dir):
    file_list = os.listdir(data_dir)
    for file in file_list:
        if os.path.splitext(file)[1] == '.txt':
            file = os.path.join(data_dir, file)
            with open(file, 'r', encoding='utf-8') as f:
                raw_data = f.read()
            raw_data = re.sub(r'\.\.\.', r'…', raw_data)
            raw_data = re.sub(r'(\D)\.', r'\1。', raw_data)
            raw_data = re.sub(',', '，', raw_data)
            raw_data = re.sub('!', '！', raw_data)
            raw_data = re.sub(r'\?', '？', raw_data)
            raw_data = re.sub(';', '；', raw_data)
            raw_data = re.sub(':', '：', raw_data)
            raw_data = re.sub('--', '——', raw_data)
            raw_data = re.sub(r'(\d+月\d+日)(星期\w)(\w+)', r'\1 \2 \3', raw_data)
            with open(file, 'w', encoding='utf-8') as f:
                f.write(raw_data)

def txt_to_text(data_dir, num_steps):
    file_list = os.listdir(data_dir)
    data = []
    for file in file_list:
        if os.path.splitext(file)[1] == '.txt':
            file = os.path.join(data_dir, file)
            with open(file, 'r', encoding = 'utf-8') as f:
                raw_data = f.readlines()
            for line in raw_data:
                if line == '\n':
                    continue
                if re.match('\d+月\d+日 星期\w \w+\n', line) is not None or re.match('\d+月\d+日 星期？ ？\n', line) is not None:
                    data.append('')
                elif len(data[-1] + line) <= num_steps:
                    data[-1] += line
                elif len(line) < num_steps:
                    data.append(line)
                else:
                    while len(line) > num_steps:
                        for end in ['。', '了', '！', '，', '？', '']:
                            endi = line[: num_steps].rfind(end) + 1
                            if endi != 0:
                                data.append(line[: endi])
                                line = line[endi :]
                                break
                    data.append(line)
    return data

def load_data_dairy(batch_size, num_steps):
    paragraphs = txt_to_text('./rawData/dairy', num_steps)
    tokens = [tokenize(paragraph) for paragraph in paragraphs]
    vocab = Vocabulary(tokens = tokens, reserved_tokens = ['<usr>', '<gpt>', '<eos>', '<pad>'], min_freq = 0)
    array, valid_lens = build_array(tokens, vocab, num_steps)
    dataset = data.TensorDataset(array, valid_lens)
    data_iter = data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

    print(f'num_tokens: {len([token for paragraph in paragraphs for token in paragraph])}\nvocab_size: {len(vocab)}')
    return data_iter, vocab

if __name__ == '__main__':
    # preprocess('../rawData/dairy')
    print('')