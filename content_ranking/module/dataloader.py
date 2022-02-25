import json
import os
import pickle
from pathlib import Path
from random import shuffle

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import trange

from common.module.vocabulary import Vocab, PAD_TOKEN, UNK_TOKEN


class ContentRankingDataset(Dataset):
    def __init__(self, vocab: Vocab, title_max_words, max_words, max_sents, input_dir, label_dir, debug=False):
        self.vocab = vocab
        self.title_max_words = title_max_words
        self.max_words = max_words
        self.max_sents = max_sents
        self.PAD_id = vocab.token2id[PAD_TOKEN]
        self.UNK_id = vocab.token2id[UNK_TOKEN]

        self.input_dir = input_dir
        self.label_dir = label_dir
        self._filenames = [filename for filename in os.listdir(self.input_dir) if filename.endswith('.json')]
        self._filenames.sort()

        if debug:
            self._filenames = self._filenames[:10000]

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        filename = self._filenames[idx]

        input_path = os.path.join(self.input_dir, filename)
        assert os.path.exists(input_path)
        with open(input_path) as f1:
            input_data = json.load(f1)

        label_path = os.path.join(self.label_dir, filename)
        assert os.path.exists(label_path)
        with open(label_path) as f2:
            label_data = json.load(f2)

        cache = dict()
        cache['id'] = input_data['id']

        title = input_data['title']
        title_words = title.split()[:self.title_max_words]
        title_input_ids = self.vocab[title_words]
        cache['title_input_ids'] = title_input_ids

        section = input_data['section'][:self.max_sents]
        section_inputs_ids = []
        num_words = 0
        for sent in section:
            sent_words = sent.split()[:self.max_words]
            sent_input_ids = self.vocab[sent_words]
            section_inputs_ids.append(sent_input_ids)
            num_words = max(num_words, len(sent_input_ids))
        cache['section_input_ids'] = section_inputs_ids

        cache['sect_rouge'] = label_data['sect_rouge']
        cache['sent_rouges'] = label_data['sent_rouges'][:self.max_sents]

        cache['title_num_words'] = len(title_input_ids)
        cache['num_sents'] = len(section_inputs_ids)
        cache['num_words'] = num_words

        return cache

    def getkeys(self):
        keys_path = os.path.join(self.input_dir, 'keys.pickle')
        if os.path.exists(keys_path):
            with open(keys_path, 'rb') as f:
                keys = pickle.load(f)
            print(f'Load keys from {keys_path}')
        else:
            keys = []
            for i in trange(len(self)):
                data = self.__getitem__(i)
                section_input_ids = data['section_input_ids']
                num_sent = len(section_input_ids)
                num_word = max([len(sent_input_ids) for sent_input_ids in section_input_ids])
                keys.append((num_sent, num_word))
            with open(keys_path, 'wb') as f:
                pickle.dump(keys, f)
            print(f'Keys saved in {keys_path}')
        return keys


class BucketSampler(Sampler):
    def __init__(self, data_source: ContentRankingDataset, bucket_size, batch_size):
        super(BucketSampler, self).__init__(data_source)
        assert bucket_size % bucket_size == 0
        self.size = len(data_source)
        self.keys = data_source.getkeys()
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.num_bucket = (self.size + self.bucket_size - 1) // bucket_size

    def __len__(self):
        return self.size

    def __iter__(self):
        indices = [i for i in range(self.size)]
        shuffle(indices)
        bucket_size, batch_size, num_bucket = self.bucket_size, self.batch_size, self.num_bucket
        sorted_indices = []
        for i in range(num_bucket):
            if i == num_bucket - 1:
                chunk = indices[bucket_size * i:]
            else:
                chunk = indices[bucket_size * i: bucket_size * (i + 1)]
            chunk.sort(key=lambda i: self.keys[i])
            batches = []
            for j in range(0, len(chunk), batch_size):
                batches.append(chunk[j:j + batch_size])
            shuffle(batches)
            for batch in batches:
                sorted_indices.extend(batch)
        return iter(sorted_indices)


def pad_input_ids(data, cache, pad_idx=0):
    batch = len(data)
    batch_title_input_ids = [d['title_input_ids'] for d in data]
    batch_section_input_ids = [d['section_input_ids'] for d in data]
    batch_sent_rouges = [d['sent_rouges'] for d in data]

    title_max_words = max([d['title_num_words'] for d in data])
    max_sents = max([d['num_sents'] for d in data])
    max_words = max([d['num_words'] for d in data])

    padded_title_input_ids = torch.full((batch, title_max_words), pad_idx)
    padded_section_input_ids = torch.full((batch, max_sents, max_words), pad_idx)
    padded_sent_rouges = torch.zeros((batch, max_sents))
    for i in range(batch):
        title_input_ids = batch_title_input_ids[i]
        section_input_ids = batch_section_input_ids[i]
        sent_rouges = batch_sent_rouges[i]

        padded_title_input_ids[i, :len(title_input_ids)] = torch.tensor(title_input_ids, dtype=torch.long)
        for j in range(len(section_input_ids)):
            sent_input_ids = section_input_ids[j]
            padded_section_input_ids[i, j, :len(sent_input_ids)] = torch.tensor(sent_input_ids, dtype=torch.long)
        padded_sent_rouges[i, :len(sent_rouges)] = torch.tensor(sent_rouges, dtype=torch.float)

    cache['title_input_ids'] = padded_title_input_ids
    cache['section_input_ids'] = padded_section_input_ids
    cache['sent_rouges'] = padded_sent_rouges


def collate_fn(data):
    cache = dict()
    cache['id'] = [d['id'] for d in data]
    cache['sect_rouge'] = torch.tensor([d['sect_rouge'] for d in data], dtype=torch.float)
    pad_input_ids(data, cache)
    return cache
