from __future__ import division
import argparse
import pandas as pd

# useful stuff
from tqdm import tqdm
from collections import Counter
from itertools import islice
import numpy as np
import re
import operator
from scipy.spatial.distance import cosine
from nltk.tokenize import TweetTokenizer
import pickle

min_freq = 5  # retain the words appearing at least this number of times
oov_token = 0  # for out-of-vocabulary words

__authors__ = ['author1', 'author2', 'author3']
__emails__ = ['fatherchristmoas@northpole.dk', 'toothfairy@blackforest.no', 'easterbunny@greenfield.de']

tokenizer = TweetTokenizer()


def preprocess_text(text):
    text = text.lower()  # Lower everything
    text = re.sub(' +', ' ', text)  # Stop extra white space
    text = text.strip()  # strip leading and trailing white space
    return text


def text2sentences(path):
    sentences = []
    with open(path) as f:
        for line in f:
            line = preprocess_text(line)
            tokens = tokenizer.tokenize(line)
            sentences.append(tokens)
    return sentences


class SkipGramPreprocessing:
    def __init__(self, path=None, path_save=None):
        if path is not None:
            self.cleaned_text = text2sentences(path)
            self.tokens = []
            for review in self.cleaned_text:  # Create list "tokens" containing all the tokens of cleaned_text
                self.tokens += review
        self.word2int = None
        self.int2word = None
        self.counts = None
        self.text2int = None
        self.path_save = path_save

    def generate_utils(self):
        self.generate_counts()
        self.generate_vocab()
        self.texts2indexes()
        # Saving vocab
        with open(self.path_save + 'vocab.file', 'wb') as voc:
            pickle.dump(self.word2int, voc, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path_save + 'inv_vocab.file', 'wb') as inv_voc:
            pickle.dump(self.int2word, inv_voc, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path_save + 'counts.file', 'wb') as counts:
            pickle.dump(self.counts, counts, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_counts(self):
        counts = dict(Counter(self.tokens))
        counts = {word: num for num, word in enumerate(counts) if num >= min_freq}
        self.counts = counts

    def generate_vocab(self):
        sorted_counts = sorted(self.counts.items(), key=operator.itemgetter(1), reverse=True)
        # We will assign to each word an index based on its frequency in the corpus,
        # the most frequent word will get index equal to 1 and 0 is reserved for out-of-vocabulary words
        word_to_index = dict([(_tuple[0], idx) for idx, _tuple in enumerate(sorted_counts, 1)])
        # Inverse of word to index
        index_to_word = {v: k for k, v in word_to_index.items()}
        self.word2int, self.int2word = word_to_index, index_to_word
        pass

    def texts2indexes(self):
        text_2_ints = []
        # transform each sentence into a list of word indexes
        for i, txt in enumerate(self.cleaned_text):
            sublist = self.text2indexes(txt)
            text_2_ints.append(sublist)
        self.text2int = text_2_ints

    def text2indexes(self, txt):
        sublist = []
        for token in txt:
            idx = self.word2indexes(token)
            sublist.append(idx)
        return sublist

    def word2indexes(self, word):
        try:
            idx = int(self.word2int[word])
        except KeyError:
            idx = int(oov_token)
        return idx

    def load(self):
        with open(self.path_save + 'vocab.file', 'rb') as voc:
            self.word2int = pickle.load(voc)
        with open(self.path_save + 'inv_vocab.file', 'rb') as inv_voc:
            self.int2word = pickle.load(inv_voc)
        with open(self.path_save + 'counts.file', 'rb') as counts:
            self.counts = pickle.load(counts)


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, skipgram_datas, nEmbed=128, negativeRate=5, winSize=5, num_epochs=20, lr=0.05, decay=1e-6,
                 save_freq=1, path_save=None, train=True):
        self.skipgram_preprocess = skipgram_datas
        self.max_window_size = winSize  # extends on both sides of the target word
        self.n_windows = int(1e6)  # number of windows to sample at each epoch
        self.n_negs = negativeRate  # number of negative examples to sample for each positive
        self.dim_embed = nEmbed  # dimension of the embedding space
        self.vocab = skipgram_datas.word2int  # Vocab (Word to Index)
        self.vocab_inv = skipgram_datas.int2word  # Inverse Vocabulary (Index to Word)
        if train:
            self.docs = skipgram_datas.text2int  # Mapping document to list of integers
        self.counts = skipgram_datas.counts
        self.token_ints = range(1, len(self.vocab) + 1)
        neg_distr = np.sqrt([self.counts[self.vocab_inv[idx]] for idx in self.token_ints])
        self.neg_distr = neg_distr / sum(neg_distr)  # For normalization
        self.Wt = np.random.normal(size=(len(self.vocab) + 1, self.dim_embed))
        self.Wc = np.random.normal(size=(len(self.vocab) + 1, self.dim_embed))
        self.n_epochs = num_epochs
        self.lr = lr
        self.decay = decay
        self.total_its = 0
        self.path_save = path_save
        self.save_freq = save_freq  # Number of epoch before each checkpoint (saving Wt & Wc)

    def sample(self, docs, max_window_size, n_windows):
        """generate target,context pairs and negative examples"""
        windows = []
        for i, doc in enumerate(docs):
            windows.append(list(self.get_windows(doc, 2 * np.random.randint(1, max_window_size) + 1)))
        windows = [elt for sublist in windows for elt in sublist]
        windows = list(np.random.choice(windows, size=n_windows))
        all_negs = list(np.random.choice(self.token_ints, size=self.n_negs * len(windows), p=self.neg_distr))
        return windows, all_negs

    def train(self):
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        # Sample all training examples once for all (instead of doing it at each iteration, saving time ??)
        windows, all_negs = self.sample(self.docs, self.max_window_size, self.n_windows)
        # Random
        np.random.shuffle(windows)

        total_loss = 0
        with tqdm(total=len(windows), unit_scale=True, postfix={'loss': 0.0, 'lr': self.lr},
                  desc="Epoch : %i/%i" % (epoch + 1, self.n_epochs), ncols=50) as pbar:
            for counter, sentence in enumerate(windows):
                target = sentence[int(len(sentence) / 2)]  # retrieve the target at the center
                pos = list(sentence)
                del pos[int(len(sentence) / 2)]  # remove the target word (the one that we want to predict)

                negs = all_negs[self.n_negs * counter:self.n_negs * counter + self.n_negs]

                prods = self.Wc[pos + negs, ] @ self.Wt[target, ]
                prodpos = prods[0:len(pos), ]
                prodnegs = prods[len(pos):(len(pos) + len(negs)), ]

                partials_pos, partials_negs, partial_target = self.compute_gradients(pos, negs, target, prodpos, prodnegs)

                lr = self.lr * 1 / (1 + self.decay * self.total_its)
                self.total_its += 1

                # Update Params
                self.Wt[target, ] -= lr * partial_target
                self.Wc[pos, ] -= partials_pos * lr
                self.Wc[negs, ] -= partials_negs * lr

                total_loss += self.compute_loss(prodpos, prodnegs)

                pbar.set_postfix({'loss': total_loss / (counter + 1), 'lr': lr})
                pbar.update(1)
                if epoch % self.save_freq == 0:
                    self.save()

    def compute_gradients(self, pos, negs, target, prodpos, prodnegs):
        factors_pos = 1 / (np.exp(prodpos) + 1)
        factors_negs = 1 / (np.exp(-prodnegs) + 1)

        partial_pos = np.array([factors_pos[k, ] * -self.Wt[target, ] for k in range(len(factors_pos))])
        partial_negs = np.array([factors_negs[k, ] * self.Wt[target, ] for k in range(len(factors_negs))])

        term_pos = - self.Wc[pos, ].T @ factors_pos
        term_negs = self.Wc[negs, ].T @ factors_negs
        partial_target = np.sum(term_pos, axis=0) + np.sum(term_negs, axis=0)
        return partial_pos, partial_negs, partial_target

    def save(self):
        np.save(self.path_save + 'input_vecs', self.Wt, allow_pickle=False)
        np.save(self.path_save + 'output_vecs', self.Wc, allow_pickle=False)

    def similarity(self, word1, word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        word1, word2 = preprocess_text(word1), preprocess_text(word2)
        idx1, idx2 = self.skipgram_preprocess.word2indexes(word1), self.skipgram_preprocess.word2indexes(word2)
        emb_1, emb_2 = self.Wt[idx1, ].reshape(1, -1), self.Wt[idx2, ].reshape(1, -1)
        cos_similarity = cosine(emb_1, emb_2)
        return round(float(cos_similarity), 4)

    def load(self, path):
        self.Wt = np.load(path + 'input_vecs.npy')
        self.Wc = np.load(path + 'output_vecs.npy')

    @staticmethod
    def get_windows(seq, n):
        """
        returns a sliding window (of width n) over data from the iterable
        taken from: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator/6822773#6822773
        """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    @staticmethod
    def compute_loss(prodpos, prodnegs):
        """prodpos and prodnegs are numpy vectors containing the dot products of the context word vectors with
        the target word vector"""
        term_pos, term_negs = np.log(1 + np.exp(-prodpos)), np.log(1 + np.exp(prodnegs))
        return np.sum(term_pos) + np.sum(term_negs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()
    if not opts.test:
        skipgram_utils = SkipGramPreprocessing(opts.text, opts.model)
        skipgram_utils.generate_utils()
        sg = SkipGram(skipgram_utils, path_save=opts.model)
        sg.train()
        sg.save()
    else:
        pairs = loadPairs(opts.text)
        skipgram_utils = SkipGramPreprocessing(path_save=opts.model)
        skipgram_utils.load()
        sg = SkipGram(skipgram_datas=skipgram_utils, path_save=opts.model, train=False)
        sg.load(opts.model)
        for a, b, _ in pairs:
            print(sg.similarity(a, b))
