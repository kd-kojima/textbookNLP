# -*- coding: utf-8 -*-
# pre-processing

import collections
import itertools
import pandas as pd

import matplotlib.pyplot as plt

import MeCab
import neologdn
import unicodedata

plt.rcParams['font.family'] = 'IPAexGothic'

class RawText():
    def __init__(self, filepath=None, text=None):
        if text is not None:
            self.text = text
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.text = f.readlines()
        self.text = ''.join(self.text).replace('\n', '')
        self.count = None
        self.freq_words = None
        self.stopwords = None
    
    def get_sentences(self):
        return self.text.split('。')

    def normalize(self):
        self.text = neologdn.normalize(self.text)
        self.text = unicodedata.normalize('NFKC', self.text)

    def count_words(self, mecab=MeCab.Tagger('-Owakati'), most=50):
        words = mecab.parse(self.text).split()
        self.count = collections.Counter(words).most_common()
        pd.set_option('display.max_rows', most)
        print(pd.DataFrame([list(r) for r in list(self.count[:most])]))

        y = [c for w, c in self.count]
        x = [w for w, c in self.count]
        plt.bar(x[0:most], y[0:most])
        plt.xticks(rotation=90)
        plt.show()

        self.freq_words = x

    def set_stopwords(self):
        print('Please input number of word than which more frequent words you add stopwords.')
        max = int(input())
        self.stopwords = self.freq_words[:max]

    def get_count(self):
        if self.count is None:
            self.count_words()
        return self.count

    def get_freq_words(self):
        if self.freq_words is None:
            self.count_words()
        return self.freq_words

    def get_stopwords(self):
        if self.stopwords is None:
            self.set_stopwords()
        return self.stopwords

    def print(self):
        print(self.text)

class Sentences():
    def __init__(self, rawtext: RawText, stopwords: list = None):
        self.sts = rawtext.get_sentences()
        if stopwords is None:
            self.stopwords = rawtext.get_stopwords()
        else:
            self.stopwords = stopwords

    def normalize(self):
        for st in self.sts:
            st = neologdn.normalize(st)
            st = unicodedata.normalize('NFKC', st)
        
    def get_noun(self, mecab=MeCab.Tagger('-Odump')):
        nouns = []
        for st in self.sts:
            morph = [x.split(' ') for x in mecab.parse(st).replace(',', ' ').split('\n')][:-1]
            nouns.append([x[1] for x in morph if '名詞' in x[2] and x[1] not in self.stopwords])
        return nouns

    def print(self):
        for st in self.sts:
            print(st)
        
class NounSentences():
    def __init__(self, sentences: Sentences):
        self.nouns = sentences.get_noun()

    def print(self):
        for st in self.nouns:
            print(st)

    def make_comb(self):
        comb_ns = [list(itertools.combinations(st, 2)) for st in self.nouns]
        comb = list(itertools.chain.from_iterable([[tuple(sorted(c)) for c in comb] for comb in comb_ns]))
        return comb

class Combinations():
    def __init__(self, nouns: NounSentences):
        self.combs = nouns.make_comb()
