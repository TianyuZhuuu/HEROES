import json
import os
from pathlib import Path

import dgl
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset

from common.module.vocabulary import PAD_TOKEN, UNK_TOKEN

stopwords = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']


class SummarizationDataset(Dataset):
    def __init__(self, vocab,
                 title_max_num_words, max_num_sects, max_num_sents, max_num_words, max_dist,
                 input_dir, tfidf_path, label_dir,
                 use_sect_nodes=True, use_local_sent_conns=True, use_global_sent_conns=False):
        super(SummarizationDataset, self).__init__()
        self.vocab = vocab

        self.title_max_num_words = title_max_num_words
        self.max_num_sects = max_num_sects
        self.max_num_sents = max_num_sents
        self.max_num_words = max_num_words
        self.max_dist = max_dist

        self.pad_id = vocab[PAD_TOKEN]
        self.filter_wids = set()
        # Add stopwords
        for w in stopwords:
            if vocab[w] != vocab[UNK_TOKEN]:
                self.filter_wids.add(vocab[w])
        # Add punctuations
        for p in punctuations:
            if vocab[p] != vocab[UNK_TOKEN]:
                self.filter_wids.add(vocab[p])
        # Add low tf-idf words
        low_tfidf_count = 0
        with open(tfidf_path) as f:
            for line in f:
                word = line.strip()
                if vocab[word] != vocab[UNK_TOKEN]:
                    self.filter_wids.add(vocab[word])
                    low_tfidf_count += 1
                    if low_tfidf_count >= 5000:
                        break
        # Add [PAD] token
        self.filter_wids.add(self.pad_id)

        self.input_dir = input_dir
        self.label_dir = label_dir

        self._input_paths = list(Path(input_dir).glob('*.json'))
        self._input_paths.sort()

        self.use_sect_nodes = use_sect_nodes
        self.use_local_sent_conns = use_local_sent_conns
        self.use_global_sent_conns = use_global_sent_conns

    def __len__(self):
        return len(self._input_paths)

    def __getitem__(self, idx):
        input_path = self._input_paths[idx]

        with open(input_path) as f:
            input_data = json.load(f)

        # input
        id = input_data['id']
        reference = input_data['reference']
        titles = input_data['titles']
        sections = input_data['sections']

        valid_sect_num = 0
        valid_sent_nums = []

        padded_title_input_ids = torch.zeros((self.max_num_sects, self.title_max_num_words), dtype=torch.long)
        padded_input_ids = torch.zeros((self.max_num_sects, self.max_num_sents, self.max_num_words), dtype=torch.long)
        padded_start_dists = torch.zeros((self.max_num_sects, self.max_num_sents), dtype=torch.long)
        padded_end_dists = torch.zeros((self.max_num_sects, self.max_num_sents), dtype=torch.long)

        for sect_idx, sect in enumerate(sections):
            if sect_idx == self.max_num_sects:
                break

            # a valid section
            valid_sect_num += 1
            # the number of valid sentences in this section
            valid_sent_nums.append(len(sect))

            title = titles[sect_idx]
            title_words = title.split()[:self.title_max_num_words]
            title_ids = self.vocab[title_words]
            padded_title_input_ids[sect_idx, :len(title_ids)] = torch.tensor(title_ids, dtype=torch.long)

            for sent_idx, sent_info in enumerate(sections[sect_idx]):
                sent = sent_info['text']
                sent_words = sent.split()[:self.max_num_words]
                sent_ids = self.vocab[sent_words]
                padded_input_ids[sect_idx, sent_idx, :len(sent_ids)] = torch.tensor(sent_ids, dtype=torch.long)
                padded_start_dists[sect_idx, sent_idx] = sent_info['start_dist'] + 1
                padded_end_dists[sect_idx, sent_idx] = sent_info['end_dist'] + 1

        padded_start_dists = torch.clamp(padded_start_dists, max=self.max_dist + 1)
        padded_end_dists = torch.clamp(padded_end_dists, max=self.max_dist + 1)

        # label
        label_path = os.path.join(self.label_dir, f'{id}.json')
        with open(label_path) as f:
            label_data = json.load(f)
        labels = label_data['labels']

        padded_labels = torch.zeros((self.max_num_sects, self.max_num_sents), dtype=torch.float)
        for sect_idx in range(len(labels)):
            if sect_idx == self.max_num_sects:
                break
            sect_labels = labels[sect_idx][:self.max_num_sents]
            padded_labels[sect_idx, :len(sect_labels)] = torch.tensor(sect_labels, dtype=torch.float)

        graph = self.build_graph(padded_input_ids.numpy(), valid_sect_num, valid_sent_nums, padded_start_dists,
                                 padded_end_dists)

        cache = {'id': id, 'reference': reference,
                 'sections': [[sent_info['text'] for sent_info in section] for section in sections],
                 'title_input_ids': padded_title_input_ids, 'input_ids': padded_input_ids,
                 'labels': padded_labels, 'graph': graph}
        return cache

    def build_graph(self, input_ids, valid_num_sects, valid_num_sents_list, padded_start_dists, padded_end_dists):
        # wid: word idx in the vocabulary
        # wnid: word node idx
        wid2wnid, wnid2wid = {}, {}
        num_word_nodes, pad_wnid = 0, -1

        # Construct wid-wnid mappings
        for sect_wids in input_ids:
            for sent_wids in sect_wids:
                for wid in sent_wids:
                    if wid not in self.filter_wids and wid not in wid2wnid.keys():
                        wid2wnid[wid] = num_word_nodes
                        wnid2wid[num_word_nodes] = wid
                        num_word_nodes += 1

        # Initialize PAD word node
        if sum(valid_num_sents_list) < self.max_num_sects * self.max_num_sents:
            wid2wnid[self.pad_id] = pad_wnid = num_word_nodes
            wnid2wid[num_word_nodes] = self.pad_id
            num_word_nodes += 1

        # set of all valid sentence node ids
        global_valid_snid_set = set()
        for sect_idx in range(valid_num_sects):
            valid_num_sents = valid_num_sents_list[sect_idx]
            for sent_idx in range(valid_num_sents):
                snid = sect_idx * self.max_num_sents + sent_idx
                global_valid_snid_set.add(snid)

        sent_word_conns = []
        word_sent_conns = []
        sent_sent_conns = []
        sect_sent_conns = []
        sent_sect_conns = []
        sect_sect_conns = []

        for sect_idx in range(self.max_num_sects):
            if sect_idx < valid_num_sects:
                # a valid section

                sect = input_ids[sect_idx]
                valid_num_sents = valid_num_sents_list[sect_idx]

                # valid sentence node id set within the section
                local_valid_snid_set = set()
                for sent_idx in range(valid_num_sents):
                    snid = sect_idx * self.max_num_sents + sent_idx
                    local_valid_snid_set.add(snid)

                for sent_idx in range(self.max_num_sents):
                    # sentence node id
                    snid = sect_idx * self.max_num_sents + sent_idx

                    if sent_idx < valid_num_sents:
                        # a valid sentence
                        sent = sect[sent_idx]

                        has_word = False

                        # word-sentence connection
                        for wid in set(sent):
                            if wid in wid2wnid:
                                wnid = wid2wnid[wid]
                                sent_word_conns.append((snid, wnid))
                                word_sent_conns.append((wnid, snid))
                                has_word = True

                        if not has_word:
                            # All words are filtered, use PAD token
                            # If not initialized
                            if pad_wnid == -1:
                                wid2wnid[self.pad_id] = pad_wnid = num_word_nodes
                                wnid2wid[num_word_nodes] = self.pad_id
                                num_word_nodes += 1
                            sent_word_conns.append((snid, pad_wnid))
                            word_sent_conns.append((pad_wnid, snid))

                        # sentence-sentence connection
                        if self.use_local_sent_conns:
                            for other_snid in local_valid_snid_set:
                                sent_sent_conns.append((other_snid, snid))
                        if self.use_global_sent_conns:
                            for other_snid in global_valid_snid_set:
                                sent_sent_conns.append((other_snid, snid))

                        if self.use_sect_nodes:
                            # section-sentence connection
                            for other_sect_idx in range(valid_num_sects):
                                sect_sent_conns.append((other_sect_idx, snid))

                            # sentence-section connection
                            sent_sect_conns.append((snid, sect_idx))
                    else:
                        # not a valid sentence

                        # word-sentence connection
                        # let the PAD token be the only word for PAD sentence
                        word_sent_conns.append((pad_wnid, snid))
                        sent_word_conns.append((snid, pad_wnid))

                        # sentence-sentence connection
                        if self.use_local_sent_conns or self.use_global_sent_conns:
                            sent_sent_conns.append((snid, snid))

                        if self.use_sect_nodes:
                            sect_sent_conns.append((sect_idx, snid))

                if self.use_sect_nodes:
                    # section-section connection
                    for other_sect_idx in range(valid_num_sects):
                        sect_sect_conns.append((other_sect_idx, sect_idx))

            else:
                # not a valid section

                for sent_idx in range(self.max_num_sents):
                    snid = sect_idx * self.max_num_sents + sent_idx

                    # word-sentence connection
                    # let the PAD token be the only word for PAD sentence
                    word_sent_conns.append((pad_wnid, snid))
                    sent_word_conns.append((snid, pad_wnid))

                    # sentence-sentence connection
                    if self.use_local_sent_conns or self.use_global_sent_conns:
                        sent_sent_conns.append((snid, snid))

                    if self.use_sect_nodes:
                        # section-sentence connection
                        sect_sent_conns.append((sect_idx, snid))

                        # sentence-section connection
                        sent_sect_conns.append((snid, sect_idx))

                if self.use_sect_nodes:
                    # section-section connection
                    for other_sect_idx in range(valid_num_sects):
                        sect_sect_conns.append((other_sect_idx, sect_idx))

        num_nodes_dict = {
            'word': num_word_nodes,
            'sent': self.max_num_sects * self.max_num_sents,
        }
        graph_data = {
            ('sent', 'sent-to-word', 'word'): tuple(zip(*sent_word_conns)),
            ('word', 'word-to-sent', 'sent'): tuple(zip(*word_sent_conns)),
        }

        if len(sent_sent_conns) != 0:
            graph_data[('sent', 'sent-to-sent', 'sent')] = tuple(zip(*sent_sent_conns))

        if self.use_sect_nodes:
            num_nodes_dict['sect'] = self.max_num_sects
            graph_data[('sect', 'sect-to-sent', 'sent')] = tuple(zip(*sect_sent_conns))
            graph_data[('sent', 'sent-to-sect', 'sect')] = tuple(zip(*sent_sect_conns))
            graph_data[('sect', 'sect-to-sect', 'sect')] = tuple(zip(*sect_sect_conns))

        g = dgl.heterograph(graph_data, num_nodes_dict, idtype=torch.int32)

        g.nodes['word'].data['id'] = torch.LongTensor(list(wnid2wid.values()))
        g.nodes['sent'].data['start_dist'] = torch.reshape(padded_start_dists, (-1,))
        g.nodes['sent'].data['end_dist'] = torch.reshape(padded_end_dists, (-1,))

        return g


def collate_fn(instances):
    res = {
        'id': [inst['id'] for inst in instances],
        'reference': [inst['reference'] for inst in instances],
        'sections': [inst['sections'] for inst in instances],
        'title_input_ids': torch.stack([inst['title_input_ids'] for inst in instances]),
        'input_ids': torch.stack([inst['input_ids'] for inst in instances]),
        'labels': torch.stack([inst['labels'] for inst in instances]),
        'graph': dgl.batch([inst['graph'] for inst in instances])
    }
    return res
