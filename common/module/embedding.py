import numpy as np
import torch

from common.module.vocabulary import Vocab

# From https://github.com/dqwang122/HeterSumGraph
# PE(pos, 2i) = sin(pos / (10000 ^ (2i / d_model))
# PE(pos, 2i + 1) = cos(pos / 10000 ^ (2i / d_model))
def get_sinusoid_encoding_table(max_seq_len, d_model, padding_idx=None):
    def calc_angle(pos, idx):
        return pos / np.power(10000, 2 * (idx // 2) / d_model)

    def get_pos_angle_vec(pos):
        return [calc_angle(pos, idx) for idx in range(d_model)]

    sinusoid_table = np.array([get_pos_angle_vec(pos) for pos in range(max_seq_len)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.tensor(sinusoid_table, dtype=torch.float)


class WordEmbedding:
    def __init__(self, path, vocab: Vocab):
        self.path = path
        self.vocablist = set(vocab.token2id)
        self.vocab = vocab

    def load_word_vctrs(self):
        word_vctrs = {}
        dim = -1
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                word, values_str = line.split(' ', maxsplit=1)
                if word in self.vocablist:
                    vector = [float(val) for val in values_str.split()]
                    dim = len(vector)
                    word_vctrs[word] = vector
        vctrs = []
        for word in word_vctrs:
            vctrs.append(word_vctrs[word])
        unk_vctr = torch.tensor(vctrs, dtype=torch.float).mean(dim=0)

        embedding = torch.zeros((len(self.vocab), dim), dtype=torch.float)
        hits = 0
        oov = 0
        for i in range(len(self.vocab)):
            word = self.vocab.id2token[i]
            if word not in word_vctrs:
                oov += 1
                embedding[i, :] = unk_vctr
            else:
                hits += 1
                embedding[i, :] = torch.tensor(word_vctrs[word], dtype=torch.float)
        print(f'Vocabulary hit ratio = {hits / (hits + oov):.2%}')
        return embedding
