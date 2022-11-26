# -*- coding: utf-8 -*-

#
# Kojima, K. (2023)
#   Co-occurrence network analysis of course guidelines for senior thesis
#

# ----------------------
# Modules
# 
import itertools, json
import pandas as pd

import MeCab
import neologdn, unicodedata

# ----------------------
# Variables
# 
filename = 'integrated' # which text data to use

IPA_DIC = '/opt/homebrew/lib/mecab/dic/ipadic'
USER_DIC = '/opt/homebrew/lib/mecab/dic/user/user.dic'

stopwords_path = './stopwords.txt'

raw_dir = './rawdata'
text_dir = './textdata'
noun_dir = './noundata'
result_dir = './result'

# ----------------------
# Read stopwords
# 
stopwords = []
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')

# ----------------------
# Read rawdata > split passage (or sentence)
# 
rawdata = ''
with open(f'{raw_dir}/{filename}.txt', 'r', encoding='utf-8') as f:
    rawdata = f.readlines()
rawdata = ''.join(rawdata).replace('\n\u3000', '<newpassage>').replace('\n', '').replace('<newpassage>', '\n')
# rawdata = ''.join(rawdata).replace('\n', '').replace('。','。\n') # split sentence
rawdata = neologdn.normalize(rawdata)
rawdata = unicodedata.normalize('NFKC', rawdata)

with open(f'{text_dir}/{filename}.txt', 'w', encoding='utf-8') as f:
    f.write(rawdata)

# ----------------------
# Read textdata > get noun, show frequency
# 
textdata = []
with open(f'{text_dir}/{filename}.txt', 'r', encoding='utf-8') as f:
    textdata = f.readlines()
textdata = [line.replace('\n', '') for line in textdata]

nouns = []
mecab = MeCab.Tagger(f'-d \"{IPA_DIC}\" -u \"{USER_DIC}\" -Odump')
for line in textdata:
    morph = [x.split(' ') for x in mecab.parse(line).replace(',', '').split('\n')][:-1]
    nouns.append([x[1] for x in morph if '名詞' in x[2] and x[1] not in stopwords])

freqs = {}
for word in list(itertools.chain.from_iterable(nouns)):
    freqs[word] = freqs.get(word, 0) + 1