# -*- coding: utf-8 -*-

#
# Kojima, K. (2023)
#   Co-occurrence network analysis of course guidelines for senior thesis
#

# ----------------------
# Import modules
# 
import math, itertools
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx

import MeCab
import neologdn, unicodedata

matplotlib.use('pgf')
plt.rcParams['text.usetex'] = True 
plt.rcParams['pgf.texsystem'] = 'lualatex'
plt.rcParams['pgf.preamble'] = r'\usepackage{unicode-math}\setmainfont{IPAexGothic}\setmathfont{Fira Math}'
plt.rcParams['pgf.rcfonts'] = False

# ----------------------
# Set variables
# 
filename = 'course_guidelines' # which text data to use
MIN_WORD_RATIO = 0.02 # (word_freq / num_of_passage(sentence))
MAX_JACCARD = 0.95
MIN_JACCARD = 0.19

IPA_DIC = '/opt/homebrew/lib/mecab/dic/ipadic'
USER_DIC = '/opt/homebrew/lib/mecab/dic/user/user.dic'

stopwords_path = './stopwords.txt'

raw_dir = './rawdata'
text_dir = './textdata'
noun_dir = './noundata'
csv_dir = './csv'
result_dir = './result'

# ----------------------
# Read stopwords
# 
stopwords = []
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')

# ----------------------
# Read rawdata
# 
rawdata = ''
with open(f'{raw_dir}/{filename}.txt', 'r', encoding='utf-8') as f:
    rawdata = f.readlines()

# ----------------------
# Split passage (or sentence)
# 
# rawdata = ''.join(rawdata).replace('\n\u3000', '<newpassage>').replace('\n', '').replace('<newpassage>', '\n')
rawdata = ''.join(rawdata).replace('\n', '').replace('。','。\n') # split sentence
rawdata = neologdn.normalize(rawdata)
rawdata = unicodedata.normalize('NFKC', rawdata)

with open(f'{text_dir}/{filename}.txt', 'w', encoding='utf-8') as f:
    f.write(rawdata)

# ----------------------
# Read textdata
# 
textdata = []
with open(f'{text_dir}/{filename}.txt', 'r', encoding='utf-8') as f:
    textdata = f.readlines()

textdata = [line.replace('\n', '') for line in textdata]

# ----------------------
# Get nouns
# 
nouns = []
mecab = MeCab.Tagger(f'-d \"{IPA_DIC}\" -u \"{USER_DIC}\" -Odump')
for line in textdata:
    morph = [x.split(' ') for x in mecab.parse(line).split('\n')][:-1]
    nouns.append([x[1] for x in morph if '名詞' in x[2] and x[1] not in stopwords])

with open(f'{noun_dir}/{filename}.txt', 'w', encoding='utf-8') as f:
    for line in nouns:
        f.write(','.join(line))
        f.write('\n')

nouns = [set(line) for line in nouns]
num_of_sets = len(nouns)

# ----------------------
# Count nouns
# 
freqs = {}
for word in list(itertools.chain.from_iterable(nouns)):
    freqs[word] = freqs.get(word, 0) + 1

freqs_df = pd.DataFrame.from_dict(freqs, orient='index', columns=['freq'])

# plt.rcParams['font.family'] = 'Hiragino sans'
# plt.bar(freqs_df.sort_values('freq', ascending=False)[0:50].index.values, freqs_df.sort_values('freq', ascending=False)['freq'][0:50])
# plt.show()

# ----------------------
# Make combinations
# 
combinations = [list(itertools.combinations(set(line), 2)) for line in nouns]
combinations = [[tuple(sorted(combi)) for combi in line] for line in combinations]

# ----------------------
# Count combinations
# 
combi_freqs = {}
for combi in list(itertools.chain.from_iterable(combinations)):
    combi_freqs[combi] = combi_freqs.get(combi, 0) + 1

combi_df = pd.DataFrame([[key[0], key[1], value] for key, value in combi_freqs.items()], columns=['w1', 'w2', 'freq'])

# ----------------------
# Calculate Jaccard coef
# 
w1_freq = []
w1_ratio = []
w2_freq = []
w2_ratio = []
jaccards = []
for _, row in combi_df.iterrows():
    jaccard = 1
    union = freqs_df.loc[row.w1].freq + freqs_df.loc[row.w2].freq - row.freq
    if union != 0:
        jaccard = row.freq / union
    w1_freq.append(freqs_df.loc[row.w1].freq)
    w1_ratio.append(freqs_df.loc[row.w1].freq / num_of_sets)
    w2_freq.append(freqs_df.loc[row.w2].freq)
    w2_ratio.append(freqs_df.loc[row.w2].freq / num_of_sets)
    jaccards.append(jaccard)

# ----------------------
# Make dataframe
# 
combi_df['w1_freq'] = w1_freq
combi_df['w1_ratio'] = w1_ratio
combi_df['w2_freq'] = w2_freq
combi_df['w2_ratio'] = w2_ratio
combi_df['jaccard'] = jaccards
combi_df.sort_values('jaccard', ascending=False).to_csv(f'{csv_dir}/{filename}_df.csv')

# ----------------------
# Set graph
# 
G = nx.Graph()
G.add_nodes_from(freqs_df.index.values)
for _, row in combi_df.iterrows():
    if row.w1_ratio >= MIN_WORD_RATIO and row.w2_ratio >= MIN_WORD_RATIO and MIN_JACCARD <= row.jaccard and row.jaccard <= MAX_JACCARD:
        G.add_edge(row.w1, row.w2, weight=row.jaccard)

G.remove_nodes_from(list(nx.isolates(G)))

k = 1 / math.sqrt(len(G.nodes()))
layout = nx.spring_layout(G, k=k, iterations=30)

pr = nx.pagerank(G)
pr_values = np.array([pr[node] for node in G.nodes()])

connecteds = []
colors = []
for i, c in enumerate(nx.connected_components(G)):
    connecteds.append(c)
    colors.append(1/50 * i)

node_colors = []
for node in G.nodes():
    for i, c in enumerate(connecteds):
        if node in c:
            node_colors.append(colors[i])
            break

# ----------------------
# Plot graph
# 
plt.figure(figsize=(8, 8))

nx.draw_networkx_nodes(G, layout, node_color=node_colors, cmap=plt.cm.get_cmap('Set3'), alpha=0.7, node_size=pr_values * 30000)
edge_width = [weight * 8 for _, _, weight in G.edges(data='weight')]
nx.draw_networkx_edges(G, layout, alpha=0.4, edge_color="darkgrey", width=edge_width)

nx.draw_networkx_labels(G, layout, font_family='Hiragino sans', font_size=10, font_weight='bold')

plt.axis('off')
plt.savefig(f'{result_dir}/{filename}.pdf', dpi=300)
plt.show()