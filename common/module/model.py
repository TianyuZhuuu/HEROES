import torch
from torch import nn

from common.utils.functions import masked_softmax, masked_max, masked_mean


class Attention(nn.Module):
    def __init__(self, size, num_hiddens, dropout):
        super(Attention, self).__init__()
        self.W_k = nn.Linear(size, num_hiddens)
        self.u = nn.Linear(num_hiddens, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vectors, mask):
        """

        :param vectors: [N x seq_len x dim]
        :param mask: [N x seq_len]
        :return:
        """
        keys, values = vectors, vectors

        # keys: [N x no. key-value pairs x num_hiddens]
        keys = torch.tanh(self.W_k(keys))

        # scores: [N x 1 x no. key-value pairs]
        scores = self.u(keys).squeeze(-1).unsqueeze(1)

        # attention_weights: [N x 1 x no. key-value pairs]
        attention_weights = masked_softmax(scores, mask)

        # values: [N x no. key-value pairs x value_dimension]
        # result: [N x value_dimension]
        return torch.bmm(attention_weights, values).squeeze(1), attention_weights.squeeze(dim=1)


class NonParametricPooling(nn.Module):
    def __init__(self, strategy):
        super(NonParametricPooling, self).__init__()
        assert strategy in ['last', 'mean', 'max'], f'Cannot recognize pooling strategy {strategy}'
        self.strategy = strategy

    def forward(self, tensor, mask):
        """

        :param tensor: [batch x seq_len x dim]
        :param mask: [batch x seq_len]
        :return:
        """
        batch, seq_len, dim = tensor.shape
        # Exclusively for bidirectional RNN
        if self.strategy == 'last':
            # [batch x seq_len x 2 x dim]
            states = tensor.reshape(batch, seq_len, 2, -1)
            fwd_states = states[:, :, 0, :]
            bwd_states = states[:, :, 1, :]
            # [batch]
            lengths = torch.sum(mask, dim=-1)
            # [batch x 1 x dim]
            index = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, dim // 2) - 1
            index = index.clamp(min=0)  # avoid index < 0 error
            # [batch x dim]
            fwd_state = torch.gather(fwd_states, dim=1, index=index).squeeze(dim=1)
            bwd_state = bwd_states[:, 0, :]
            pool = torch.cat((fwd_state, bwd_state), dim=-1)
        elif self.strategy == 'mean':
            pool = masked_mean(tensor, mask.unsqueeze(-1), dim=1)
        else:
            pool = masked_max(tensor, mask.unsqueeze(-1), dim=1)
        return pool

