import torch
from torch import nn

from common.module.model import NonParametricPooling, Attention

POOLING_STRATEGIES = ['max', 'mean', 'last', 'attention']
AGGREGATION_STRATEGIES = ['concat', 'gate', 'attention']


class Pooler(nn.Module):
    def __init__(self, pooling_strategy, hidden_size):
        super(Pooler, self).__init__()
        assert pooling_strategy in ['max', 'mean', 'last',
                                    'attention'], f'Cannot recognize pooling strategy {pooling_strategy}'
        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'attention':
            self.pooling = Attention(hidden_size, hidden_size, 0.1)
        else:
            self.pooling = NonParametricPooling(pooling_strategy)

    def forward(self, tensor, mask):
        """

        :param tensor: [batch x seq_len x dim]
        :param mask: [batch x seq_len]
        :return:
        """
        return self.pooling(tensor, mask)


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, title_pooling, word_pooling, sent_pooling):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)
        self.title_enc = nn.LSTM(300, hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.title_pooler = Pooler(title_pooling, hidden_size)
        self.word_enc = nn.LSTM(300, hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.word_pooler = Pooler(word_pooling, hidden_size)
        self.sent_enc = nn.LSTM(hidden_size, hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.sent_pooler = Pooler(sent_pooling, hidden_size)

    def forward(self, title_input_ids, section_input_ids):
        """

        :param title_input_ids:   [batch x title_max_words]
        :param section_input_ids: [batch x max_sents x max_words]
        :return:
        """
        title_mask = title_input_ids.ne(0)
        word_mask = section_input_ids.ne(0)
        sent_mask = word_mask.any(dim=-1)

        batch, num_sents, num_words = section_input_ids.shape

        # [batch x num_title_word x emb_dim]
        title_word_emb = self.emb(title_input_ids)
        # [batch x num_title_word x hidden_size]
        title_word_repr, _ = self.title_enc(title_word_emb)
        # [batch x hidden_size]
        title_repr, _ = self.title_pooler(title_word_repr, title_mask)

        # [batch*num_sent x num_word]
        flat_cont_wids = section_input_ids.reshape(-1, num_words)
        # [batch*num_sent x num_word x emb_dim]
        flat_cont_word_emb = self.emb(flat_cont_wids)
        # [batch*num_sent x num_word x hidden_size]
        flat_cont_word_repr, _ = self.word_enc(flat_cont_word_emb)

        # [batch*num_sent x hidden_size]
        flat_cont_sent_emb, _ = self.word_pooler(flat_cont_word_repr, word_mask.reshape(-1, num_words))
        # [batch x num_sent x hidden_size]
        cont_sent_emb = flat_cont_sent_emb.reshape(batch, num_sents, -1)
        # [batch x num_sent x hidden_size]
        cont_sent_repr, _ = self.sent_enc(cont_sent_emb)
        # [batch x hidden_size]
        cont_repr, *extras = self.sent_pooler(cont_sent_repr, sent_mask)

        return title_repr, cont_repr, cont_sent_repr, extras


class Aggregator(nn.Module):
    def __init__(self, dim, aggregation_strategy):
        super(Aggregator, self).__init__()
        self.aggregation_strategy = aggregation_strategy
        if aggregation_strategy == 'gate':
            self.gate = nn.Linear(2 * dim, 1)
        elif aggregation_strategy == 'attention':
            self.att = Attention(dim, dim, 0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, title_repr, cont_repr):
        if self.aggregation_strategy == 'concat':
            fused_repr = torch.cat((title_repr, cont_repr), dim=-1)
        elif self.aggregation_strategy == 'gate':
            fusion_input = torch.cat((title_repr, cont_repr), dim=-1)
            fusion_weight = torch.sigmoid(self.gate(self.dropout(fusion_input)))
            fused_repr = fusion_weight * title_repr + (-fusion_weight + 1.0) * cont_repr
        else:
            stacked = torch.stack((title_repr, cont_repr), dim=1)
            fused_repr, _ = self.att(stacked, None)
        return fused_repr


class Scorer(nn.Module):
    def __init__(self, dim):
        super(Scorer, self).__init__()
        self.out = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, repr):
        return self.out(self.dropout(repr)).squeeze(dim=-1)


class ContentRankingModule(nn.Module):
    def __init__(self, embedding, hidden_size, title_pooling, word_pooling, sent_pooling, aggregation_strategy):
        super(ContentRankingModule, self).__init__()
        assert title_pooling in POOLING_STRATEGIES, f'Cannot recognize pooling strategy {title_pooling}'
        assert word_pooling in POOLING_STRATEGIES, f'Cannot recognize pooling strategy {word_pooling}'
        assert sent_pooling in POOLING_STRATEGIES, f'Cannot recognize pooling strategy {sent_pooling}'
        assert aggregation_strategy in AGGREGATION_STRATEGIES, f'Cannot recognize aggregation strategy {aggregation_strategy}'

        self.encoder = Encoder(embedding, hidden_size, title_pooling, word_pooling, sent_pooling)
        self.aggregator = Aggregator(hidden_size, aggregation_strategy)
        self.sect_scorer = Scorer(2 * hidden_size if aggregation_strategy == 'concat' else hidden_size)
        self.sent_scorer = Scorer(hidden_size)

    def forward(self, title_input_ids, section_input_ids):
        title_repr, cont_repr, sent_repr, extras = self.encoder(title_input_ids, section_input_ids)
        fused_repr = self.aggregator(title_repr, cont_repr)
        sect_score = self.sect_scorer(fused_repr)
        sent_scores = self.sent_scorer(sent_repr)
        return sect_score, sent_scores, extras
