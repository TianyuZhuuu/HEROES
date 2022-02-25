PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocab:
    def __init__(self, vocab_path, vocab_size=-1):
        self.id2token = []
        self.token2id = {}

        for w in [PAD_TOKEN, UNK_TOKEN]:
            self.id2token.append(w)
            self.token2id[w] = len(self.token2id)

        with open(vocab_path, encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()[0]
                self.id2token.append(word)
                self.token2id[word] = len(self.token2id)
                if vocab_size != -1 and len(self.id2token) == vocab_size:
                    break

    def __len__(self):
        return len(self.id2token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token2id.get(tokens, self.token2id.get(UNK_TOKEN))
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, ids):
        if not isinstance(ids, (list, tuple,)):
            return self.id2token[ids]
        return [self.to_tokens(id) for id in ids]
