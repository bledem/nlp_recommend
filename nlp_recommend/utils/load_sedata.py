from torch.utils.data import Dataset

import numpy as np
import os
import torch


class LoadSeqData(LoadData):
    """Consider data as a sequence of characters"""

    def __init__(self, n_max=None, batch_size=None, num_steps=None,
                 use_random_iter=None, use_vocab=True,  character_based=False,
                 min_freq=1):
        super().__init__(n_max)
        self.n_max = n_max
        self.data_iter_fn = LoadSeqData.seq_data_iter_random if use_random_iter \
            else LoadSeqData.seq_data_iter_sequential
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.character_based = character_based
        self.min_freq = min_freq
        self.load_corpus()
        if use_vocab:
            self.generate_vocab()
        else:
            self.vocab = None

    def remove_low_frequency(self):
        from collections import Counter
        counter = Counter(self.corpus)
        reduced_corpus = [word for word,
                          fq in counter.items() if fq >= self.min_freq]
        # update sentence
        self.corpus_sentences = [
            [w for w in s if counter[w] >= self.min_freq] for s in self.corpus_sentences]
        # remove empty lists
        self.corpus_sentences = [
            x for x in self.corpus_sentences if len(x) > 0]
        self.corpus = reduced_corpus

    def generate_vocab(self):
        from gensim import corpora
        # Remove low frequency words
        # self.remove_low_frequency()
        self.vocab = corpora.Dictionary([self.corpus])
        # self.vocab.add_documents(x)
        special_tokens = {'<unk>': 0, '<pad>': 1}
        # shift previous tokens without overwriting
        self.vocab.patch_with_special_tokens(special_tokens)

    def __iter__(self):
        iterator = self.data_iter_fn(
            self.corpus, self.batch_size, self.num_steps, self.vocab)
        iterator = map(self.apply_vocab, iterator)
        return iterator

    def apply_vocab(self, args):
        x, y = args
        X, Y = apply_voc(self.vocab, x), apply_voc(self.vocab, y)
        return torch.tensor(X), torch.tensor(Y)

    def load_corpus(self):
        df = self.load()
        text = df.tok_lem_sentence.values
        # we load all the sentences as a very long list of unique words.
        # list of all words
        self.corpus = [word for sentence in text for word in sentence]
        self.corpus_sentences = text  # list of sentences

        if self.character_based:
            self.corpus = self.to_char(self.corpus)

    @staticmethod
    def to_char(list_of_words):
        list_of_char = []
        for w in list_of_words:
            list_of_char.extend([c for c in w])
            list_of_char.extend(' ')  # add artificial space
        return list_of_char

    @staticmethod
    def seq_data_iter_random(corpus, batch_size, num_steps, vocab=None):  # @save
        """Generate a minibatch of subsequences using random sampling."""
        # Start with a random offset (inclusive of `num_steps - 1`) to partition a
        # sequence
        corpus = corpus[random.randint(0, num_steps - 1):]
        # Subtract 1 since we need to account for labels
        num_subseqs = (len(corpus) - 1) // num_steps
        # The starting indices for subsequences of length `num_steps`
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # In random sampling, the subsequences from two adjacent random
        # minibatches during iteration are not necessarily adjacent on the
        # original sequence
        random.shuffle(initial_indices)

        def data(pos):
            # Return a sequence of length `num_steps` starting from `pos`
            return corpus[pos:pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # Here, `initial_indices` contains randomized starting indices for
            # subsequences
            initial_indices_per_batch = initial_indices[i:i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            if vocab:
                X = vocab.doc2idx(X)
                Y = vocab.doc2idx(Y)
            yield X, Y

    @staticmethod
    def seq_data_iter_sequential(corpus, batch_size, num_steps, vocab=None):  # @save
        """Generate a minibatch of subsequences using sequential partitioning."""
        # Start with a random offset to partition a sequence
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = np.array([corpus[offset:offset + num_tokens]])
        Ys = np.array(corpus[offset + 1:offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i:i + num_steps]
            Y = Ys[:, i:i + num_steps]
            yield X, Y

# class LoadSentenceData(LoadSeqData):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def load_corpus(self):
#         df = self.load()
#         text = df.tok_lem_sentence.values
#         # we load all the sentences as a very long list of unique words.
#         self.corpus = text


class LoadSentenceData(LoadSeqData):
    """ Return not idx but sentences directly"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        iterator = self.data_iter_fn(
            self.corpus, self.batch_size, self.num_steps, self.vocab)
        return iterator


class LoadLabelSentenceDataloader(Dataset, LoadData):
    """ Return sentence in string or idx and associated label. """

    def __init__(self, vocab=None, max_tokens=100,  *args, **kwargs):
        super().__init__(*args, **kwaclass LoadSeqData(LoadData):
    """Consider data as a sequence of characters"""

    def __init__(self, n_max=None, batch_size=None, num_steps=None,
                 use_random_iter=None, use_vocab=True,  character_based=False,
                 min_freq=1):
        super().__init__(n_max)
        self.n_max=n_max
        self.data_iter_fn=LoadSeqData.seq_data_iter_random if use_random_iter
            else LoadSeqData.seq_data_iter_sequential
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.character_based=character_based
        self.min_freq=min_freq
        self.load_corpus()
        if use_vocab:
            self.generate_vocab()
        else:
            self.vocab=None

    def remove_low_frequency(self):
        from collections import Counter
        counter=Counter(self.corpus)
        reduced_corpus=[word for word,
                          fq in counter.items() if fq >= self.min_freq]
        # update sentence
        self.corpus_sentences=[
            [w for w in s if counter[w] >= self.min_freq] for s in self.corpus_sentences]
        # remove empty lists
        self.corpus_sentences=[
            x for x in self.corpus_sentences if len(x) > 0]
        self.corpus=reduced_corpus

    def generate_vocab(self):
        from gensim import corpora
        # Remove low frequency words
        # self.remove_low_frequency()
        self.vocab=corpora.Dictionary([self.corpus])
        # self.vocab.add_documents(x)
        special_tokens={'<unk>': 0, '<pad>': 1}
        # shift previous tokens without overwriting
        self.vocab.patch_with_special_tokens(special_tokens)

    def __iter__(self):
        iterator=self.data_iter_fn(
            self.corpus, self.batch_size, self.num_steps, self.vocab)
        iterator=map(self.apply_vocab, iterator)
        return iterator

    def apply_vocab(self, args):
        x, y=args
        X, Y=apply_voc(self.vocab, x), apply_voc(self.vocab, y)
        return torch.tensor(X), torch.tensor(Y)

    def load_corpus(self):
        df=self.load()
        text=df.tok_lem_sentence.values
        # we load all the sentences as a very long list of unique words.
        # list of all words
        self.corpus=[word for sentence in text for word in sentence]
        self.corpus_sentences=text  # list of sentences

        if self.character_based:
            self.corpus=self.to_char(self.corpus)

    @ staticmethod
    def to_char(list_of_words):
        list_of_char=[]
        for w in list_of_words:
            list_of_char.extend([c for c in w])
            list_of_char.extend(' ')  # add artificial space
        return list_of_char

    @ staticmethod
    def seq_data_iter_random(corpus, batch_size, num_steps, vocab=None):  # @save
        """Generate a minibatch of subsequences using random sampling."""
        # Start with a random offset (inclusive of `num_steps - 1`) to partition a
        # sequence
        corpus=corpus[random.randint(0, num_steps - 1):]
        # Subtract 1 since we need to account for labels
        num_subseqs=(len(corpus) - 1) // num_steps
        # The starting indices for subsequences of length `num_steps`
        initial_indices=list(range(0, num_subseqs * num_steps, num_steps))
        # In random sampling, the subsequences from two adjacent random
        # minibatches during iteration are not necessarily adjacent on the
        # original sequence
        random.shuffle(initial_indices)

        def data(pos):
            # Return a sequence of length `num_steps` starting from `pos`
            return corpus[pos:pos + num_steps]

        num_batches=num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # Here, `initial_indices` contains randomized starting indices for
            # subsequences
            initial_indices_per_batch=initial_indices[i:i + batch_size]
            X=[data(j) for j in initial_indices_per_batch]
            Y=[data(j + 1) for j in initial_indices_per_batch]
            if vocab:
                X=vocab.doc2idx(X)
                Y=vocab.doc2idx(Y)
            yield X, Y

    @ staticmethod
    def seq_data_iter_sequential(corpus, batch_size, num_steps, vocab=None):  # @save
        """Generate a minibatch of subsequences using sequential partitioning."""
        # Start with a random offset to partition a sequence
        offset=random.randint(0, num_steps)
        num_tokens=((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs=np.array([corpus[offset:offset + num_tokens]])
        Ys=np.array(corpus[offset + 1:offset + 1 + num_tokens])
        Xs, Ys=Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches=Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X=Xs[:, i:i + num_steps]
            Y=Ys[:, i:i + num_steps]
            yield X, Y

# class LoadSentenceData(LoadSeqData):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def load_corpus(self):
#         df = self.load()
#         text = df.tok_lem_sentence.values
#         # we load all the sentences as a very long list of unique words.
#         self.corpus = text


class LoadSentenceData(LoadSeqData):
    """ Return not idx but sentences directly"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        iterator=self.data_iter_fn(
            self.corpus, self.batch_size, self.num_steps, self.vocab)
        return iterator


class LoadLabelSentenceDataloader(Dataset, LoadData):
    """ Return sentence in string or idx and associated label. """

    def __init__(self, vocab=None, max_tokens=100,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab=vocab
        self.max_tokens=max_tokens
        self.load_corpus()

    def __getitem__(self, idx):
        sentence=self.corpus_sentences[idx][:self.max_tokens]
        if len(sentence) < self.max_tokens:
            sentence.extend(['<pad>']*(self.max_tokens-len(sentence)))
        label=self.label[idx]
        if self.vocab:
            sentence=self.vocab.doc2idx(sentence)
        return sentence, label

    def load_corpus(self):
        df=self.load()
        self.corpus_sentences=df.tok_lem_sentence.values
        self.label=df.label.values
        self.label2aut=self.map_author_to_label(df[['label', 'author']])

    @ staticmethod
    def map_author_to_label(df_aut_label):
        return {pair[1]: pair[0] for pair in df_aut_label.drop_duplicates().values}

    def __len__(self):
        return len(self.corpus_sentences)


if __name__ == '__main__':
    corpus=LoadSeqData(n_max=3000, batch_size=2,
                         num_steps=5, character_based=False)
    sentence_iter=LoadSentenceData(n_max=100, batch_size=2,
                                     num_steps=5, min_freq=1)

    corpus_label=LoadLabelSentenceDataloader(n_max=100, vocab=corpus.vocab)

    for i in corpus:
        print('X: \n', i[0][:3, :3], ' \n \nY (next token): \n ', i[1][:2, :2])
        break

    print(next(iter(corpus_label)))
        sentence=self.corpus_sentences[idx][:self.max_tokens]
        if len(sentence) < self.max_tokens:
            sentence.extend(['<pad>']*(self.max_tokens-len(sentence)))
        label=self.label[idx]
        if self.vocab:
            sentence=self.vocab.doc2idx(sentence)
        return sentence, label

    def load_corpus(self):
        df=self.load()
        self.corpus_sentences=df.tok_lem_sentence.values
        self.label=df.label.values
        self.label2aut=self.map_author_to_label(df[['label', 'author']])

    @ staticmethod
    def map_author_to_label(df_aut_label):
        return {pair[1]: pair[0] for pair in df_aut_label.drop_duplicates().values}

    def __len__(self):
        return len(self.corpus_sentences)


if __name__ == '__main__':
    corpus=LoadSeqData(n_max=3000, batch_size=2,
                         num_steps=5, character_based=False)
    sentence_iter=LoadSentenceData(n_max=100, batch_size=2,
                                     num_steps=5, min_freq=1)

    corpus_label=LoadLabelSentenceDataloader(n_max=100, vocab=corpus.vocab)

    for i in corpus:
        print('X: \n', i[0][:3, :3], ' \n \nY (next token): \n ', i[1][:2, :2])
        break

    print(next(iter(corpus_label)))
