import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from common.module.embedding import get_sinusoid_encoding_table
from common.module.model import Attention


class SentEncoder(nn.Module):
    def __init__(self, hps, embed):
        super(SentEncoder, self).__init__()

        self._hps = hps
        self.max_seq_len = hps.max_num_words
        embed_size = hps.word_emb_dim

        # Hyperparameters
        in_channels = embed_size
        out_channels = 50
        min_kernel_size = 2
        max_kernel_size = 7

        self.emb = embed
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(hps.max_num_words + 1, embed_size, padding_idx=0), freeze=True)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size) for kernel_size in
                                    range(min_kernel_size, max_kernel_size + 1)])
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight, gain=np.sqrt(6))

    def forward(self, input_ids):
        """

        :param input_ids: [batch x seq_len]
        :return:
        """
        device = input_ids.device

        input_word_embed = self.emb(input_ids)  # [batch x seq_len x embed_size]

        # Generate position embeddings
        non_pad_mask = input_ids.ne(0)
        seq_lens = torch.sum(non_pad_mask, dim=-1)
        positions = torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0).expand_as(input_ids).to(device)
        masked_positions = positions.masked_fill(mask=positions > seq_lens.unsqueeze(1), value=0)
        input_pos_embed = self.pos_emb(masked_positions)  # [batch x seq_len x embed_size]

        conv_input = input_word_embed + input_pos_embed  # [batch x seq_len x embed_size]
        conv_input = torch.transpose(conv_input, 1, 2)  # [batch x embed_size x seq_len]

        conv_outputs = [F.relu(conv(conv_input)).max(dim=2)[0] for conv in self.convs]  # list of [batch x 50]
        conv_output = torch.cat(conv_outputs, dim=1)  # [batch x 300]

        return conv_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output


class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.fc = nn.Linear(2 * dim, 1)

    def forward(self, x, y):
        conc = torch.cat((x, y), dim=-1)
        z = torch.sigmoid(self.fc(conc))
        fusion = z * x + (1.0 - z) * y
        return fusion


class WordGAT(nn.Module):
    def __init__(self, word_dim, sent_dim, num_heads, ffn_inner_dim, dropout=0.1):
        super(WordGAT, self).__init__()
        self.s2w_gat = GATConv((sent_dim, word_dim), int(word_dim / num_heads), num_heads, dropout, dropout)
        self.ffn = FeedForwardNetwork(word_dim, ffn_inner_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, word_repr, sent_repr):
        batch = word_repr.shape[0]
        residule = word_repr
        s2w_subg = dgl.edge_type_subgraph(graph, ['sent-to-word'])
        update = F.elu(self.s2w_gat(s2w_subg, (sent_repr, word_repr)).reshape(batch, -1))
        update = self.dropout(update)
        output = self.ffn(residule + update)
        return output


class SentGAT(nn.Module):
    def __init__(self, word_dim, sent_dim, sect_dim, num_heads, ffn_inner_dim, dropout=0.1):
        super(SentGAT, self).__init__()
        per_head_out_feats = int(sent_dim / num_heads)
        self.w2s_gat = GATConv((word_dim, sent_dim), per_head_out_feats, num_heads, dropout, dropout)
        self.s2s_gat = GATConv(sent_dim, per_head_out_feats, num_heads, dropout, dropout)
        self.S2s_gat = GATConv((sect_dim, sent_dim), per_head_out_feats, num_heads, dropout, dropout)
        self.ffn = FeedForwardNetwork(sent_dim, ffn_inner_dim, dropout)
        self.fusion1 = Fusion(sent_dim)
        self.fusion2 = Fusion(sent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, word_repr, sent_repr, sect_repr):
        batch = sent_repr.shape[0]
        residule = sent_repr
        # word-to-sent updates
        w2s_subg = dgl.edge_type_subgraph(graph, ['word-to-sent'])
        w2s_update = F.elu(self.w2s_gat(w2s_subg, (word_repr, sent_repr)).reshape(batch, -1))
        # sent-to-sent updates
        s2s_subg = dgl.edge_type_subgraph(graph, ['sent-to-sent'])
        s2s_update = F.elu(self.s2s_gat(s2s_subg, sent_repr).reshape(batch, -1))
        # sect-to-sent updates
        S2s_subg = dgl.edge_type_subgraph(graph, ['sect-to-sent'])
        S2s_update = F.elu(self.S2s_gat(S2s_subg, (sect_repr, sent_repr)).reshape(batch, -1))
        # 2 fusion operations
        wS_update = self.fusion1(w2s_update, S2s_update)
        update = self.fusion2(wS_update, s2s_update)
        update = self.dropout(update)
        output = self.ffn(residule + update)
        return output


class SectGAT(nn.Module):
    def __init__(self, sent_dim, sect_dim, num_heads, ffn_inner_dim, dropout=0.1):
        super(SectGAT, self).__init__()
        per_head_out_feats = int(sect_dim / num_heads)
        self.s2S_gat = GATConv((sent_dim, sect_dim), per_head_out_feats, num_heads, dropout, dropout)
        self.S2S_gat = GATConv(sect_dim, per_head_out_feats, num_heads, dropout, dropout)
        self.fusion = Fusion(sect_dim)
        self.ffn = FeedForwardNetwork(sect_dim, ffn_inner_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, sent_repr, sect_repr):
        batch = sect_repr.shape[0]
        residule = sect_repr
        # sent-to-sect updates
        s2S_subg = dgl.edge_type_subgraph(graph, ['sent-to-sect'])
        s2S_update = F.elu(self.s2S_gat(s2S_subg, (sent_repr, sect_repr)).reshape(batch, -1))
        # sect-to-sect updates
        S2S_subg = dgl.edge_type_subgraph(graph, ['sect-to-sect'])
        S2S_update = F.elu(self.S2S_gat(S2S_subg, sect_repr).reshape(batch, -1))
        # fusion operation
        update = self.fusion(s2S_update, S2S_update)
        update = self.dropout(update)
        output = self.ffn(residule + update)
        return output


class HEROES(nn.Module):
    def __init__(self, hps, embedding):
        super(HEROES, self).__init__()

        self._hps = hps

        word_dim = hps.word_emb_dim
        sent_dim = hps.sent_dim
        sect_dim = hps.sect_dim
        word_num_heads = hps.word_num_heads
        sent_num_heads = hps.sent_num_heads
        sect_num_heads = hps.sect_num_heads
        ffn_inner_dim = hps.ffn_inner_dim
        emb_dim = hps.word_emb_dim
        dist_emb_dim = hps.dist_emb_dim
        cnn_dim = 300
        lstm_dim = hps.lstm_dim
        dropout = hps.dropout
        n_feat_dim = hps.n_feat_dim

        self.emb = nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)
        self.start_dist_emb = nn.Embedding(hps.max_dist + 2, dist_emb_dim, padding_idx=0)
        self.end_dist_emb = nn.Embedding(hps.max_dist + 2, dist_emb_dim, padding_idx=0)

        # Graph initialization
        # Sentence Encoder
        self.sent_cnn_encoder = SentEncoder(hps, self.emb)
        self.sent_lstm_encoder = nn.LSTM(input_size=emb_dim, hidden_size=lstm_dim // 2, num_layers=2,
                                         bidirectional=True, batch_first=True, dropout=dropout)
        self.cnn_proj = nn.Linear(cnn_dim, n_feat_dim)
        self.lstm_proj = nn.Linear(lstm_dim, n_feat_dim)
        self.dist_proj = nn.Linear(2 * dist_emb_dim, n_feat_dim)
        self.sent_feat_proj = nn.Linear(3 * n_feat_dim, sent_dim, bias=False)

        # Section Encoder
        self.sect_att_encoder = Attention(sent_dim, sent_dim, dropout)
        self.sect_lstm_encoder = nn.LSTM(input_size=sent_dim, hidden_size=lstm_dim // 2, num_layers=2,
                                         bidirectional=True, batch_first=True, dropout=dropout)
        self.sect_feat_proj = nn.Linear(lstm_dim, sect_dim, bias=False)

        # GAT for each discourse units
        self.word_gat = WordGAT(word_dim, sent_dim, word_num_heads, ffn_inner_dim, dropout)
        self.sent_gat = SentGAT(word_dim, sent_dim, sect_dim, sent_num_heads, ffn_inner_dim, dropout)
        self.sect_gat = SectGAT(sent_dim, sect_dim, sect_num_heads, ffn_inner_dim, dropout)

        self.scorer = nn.Linear(sent_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def init_word_repr(self, graph):
        word_id = graph.nodes['word'].data['id']
        word_repr = self.emb(word_id)  # [num_word_nodes x emb_dim]
        return word_repr

    def init_sent_sect_repr(self, graph, input_ids):
        batch, max_num_sects, max_num_sents, max_num_words = input_ids.shape

        word_masks = input_ids.ne(0)  # [batch x max_sects x max_sents x max_words]
        sent_masks = word_masks.any(dim=-1)  # [batch x max_sects x max_sents]

        flat_input_ids = input_ids.reshape(-1, max_num_words)

        # CNN Features
        ngram_features = self.sent_cnn_encoder(flat_input_ids)
        cnn_feat = self.cnn_proj(ngram_features)
        cnn_feat = cnn_feat.reshape(-1, cnn_feat.shape[-1])
        # LSTM Features
        sent_lstm_input = ngram_features.reshape(batch * max_num_sects, max_num_sents, -1)
        sent_lstm_output, _ = self.sent_lstm_encoder(sent_lstm_input)
        lstm_feat = self.lstm_proj(sent_lstm_output)
        lstm_feat = lstm_feat.reshape(-1, lstm_feat.shape[-1])
        # Boundary Distance Features
        start_dist = graph.nodes['sent'].data['start_dist']
        end_dist = graph.nodes['sent'].data['end_dist']
        start_dist_emb = self.start_dist_emb(start_dist)
        end_dist_emb = self.end_dist_emb(end_dist)
        dist_feat = self.dist_proj(torch.cat((start_dist_emb, end_dist_emb), dim=-1))
        # Sent Features
        sent_feat = torch.cat((cnn_feat, lstm_feat, dist_feat), dim=-1)
        sent_repr = self.sent_feat_proj(sent_feat)

        # Section Features
        sect_att_input = sent_repr.reshape(batch * max_num_sects, max_num_sents, -1)
        sect_att_output, _ = self.sect_att_encoder(sect_att_input, sent_masks.reshape(-1, max_num_sents))
        sect_lstm_input = sect_att_output.reshape(batch, max_num_sects, -1)
        sect_feat, _ = self.sect_lstm_encoder(sect_lstm_input)
        sect_feat = sect_feat.reshape(-1, sect_feat.shape[-1])
        sect_repr = self.sect_feat_proj(sect_feat)  # [num_sect_nodes x hidden_size]

        return sent_repr, sect_repr

    def forward(self, graph, input_ids):
        """

        :param input_ids: [batch x max_sects x max_sents x max_words]
        :return:
        """
        batch, max_sects, max_sents, max_words = input_ids.shape

        word_repr = self.init_word_repr(graph)
        sent_repr, sect_repr = self.init_sent_sect_repr(graph, input_ids)

        for i in range(self._hps.num_iters):
            new_word_repr = self.word_gat(graph, word_repr, sent_repr)
            new_sent_repr = self.sent_gat(graph, word_repr, sent_repr, sect_repr)
            new_sect_repr = self.sect_gat(graph, sent_repr, sect_repr)
            word_repr, sent_repr, sect_repr = new_word_repr, new_sent_repr, new_sect_repr

        sent_repr = sent_repr.reshape(batch, max_sects, max_sents, -1)
        scores = self.scorer(self.dropout(sent_repr)).squeeze(-1)
        return scores
