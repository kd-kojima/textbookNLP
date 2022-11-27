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

# ----------------------
# Set variables
# 
filename = 'integrated' # which text data to use

MIN_WORD_RATIO = {  # (word_freq / num_of_passage(sentence))
    'integrated': 0.016
} 
MAX_JACCARD = {
    'integrated': 0.95
}
MIN_JACCARD = {
    'integrated': 0.24
}

LAYOUT_K_FACTOR = {
    'integrated': 1
}
LAYOUT_ITERATION = {
    'integrated': 25
}
NODE_SIZE_FACTOR = {
    'integrated': 45000
}
EDGE_WIDTH_FACTOR = {
    'integrated': 8
}

min_word_ratio = MIN_WORD_RATIO[filename]
max_jaccard = MAX_JACCARD[filename]
min_jaccard = MIN_JACCARD[filename]
layout_k_factor = LAYOUT_K_FACTOR[filename]
layout_iteration = LAYOUT_ITERATION[filename]
node_size_factor = NODE_SIZE_FACTOR[filename]
edge_width_factor = EDGE_WIDTH_FACTOR[filename]

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
# Split sentence (or passage)
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

# plt.rcParams['font.family'] = 'IPAexGothic'
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
    if row.w1_ratio >= min_word_ratio and row.w2_ratio >= min_word_ratio and min_jaccard <= row.jaccard and row.jaccard <= max_jaccard:
        G.add_edge(row.w1, row.w2, weight=row.jaccard)

G.remove_nodes_from(list(nx.isolates(G)))

connecteds = []
connected_indices = []
colors = []
for i, c in enumerate(nx.connected_components(G)):
    connecteds.append(c)
    connected_indices.append(i)
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
# plt.figure(figsize=(8, 8))

# k = layout_k_factor / math.sqrt(len(G.nodes()))
# layout = nx.spring_layout(G, k=k, iterations=layout_iteration)

# pr = nx.pagerank(G)
# pr_values = np.array([pr[node] for node in G.nodes()])
# nx.draw_networkx_nodes(G, layout, node_color=node_colors, cmap=plt.cm.get_cmap('Set3'), alpha=0.7, node_size=pr_values * node_size_factor)

# edge_width = [weight * edge_width_factor for _, _, weight in G.edges(data='weight')]
# nx.draw_networkx_edges(G, layout, alpha=0.4, edge_color="darkgrey", width=edge_width)

# nx.draw_networkx_labels(G, layout, font_family='Hiragino Sans', font_size=10, font_weight='bold')

# plt.axis('off')
# # plt.savefig(f'{result_dir}/{filename}.pdf', dpi=300)
# plt.show()

# ----------------------
# Plot for thesis
# 
matplotlib.use('pgf')
plt.rcParams['text.usetex'] = True 
plt.rcParams['pgf.texsystem'] = 'lualatex'
plt.rcParams['pgf.preamble'] = r'\usepackage{unicode-math}\setmainfont{IPAexGothic}\setmathfont{Fira Math}'
plt.rcParams['pgf.rcfonts'] = False

color_thesis = {
    'A7': '#ffd900',
    'A8': '#59b9c6',
    'A9': '#706caa',
}
nodes_thesis = {
    'A7': [],
    'A8': [],
    'A9': [],
    'others': []
}

if filename == 'integrated':
    for i, connected in enumerate(connecteds):
        if 'x' in connected and 'y' in connected and '値' in connected:
            nodes_thesis['A7'].extend([list(G.nodes()).index(node) for node in connected])
        elif '一方' in connected and '他方' in connected:
            nodes_thesis['A7'].extend([list(G.nodes()).index(node) for node in connected])
        elif '日常' in connected:
            nodes_thesis['A8'].extend([list(G.nodes()).index(node) for node in connected])
        elif '実験' in connected and '観察' in connected and '予測' in connected:
            nodes_thesis['A8'].extend([list(G.nodes()).index(node) for node in connected])
        elif 'グラフ' in connected and '式' in connected and '対応' in connected:
            nodes_thesis['A9'].extend([list(G.nodes()).index(node) for node in connected])
        elif '座標' in connected and '平面' in connected:
            nodes_thesis['A9'].extend([list(G.nodes()).index(node) for node in connected])
        else:
            nodes_thesis['others'].extend([list(G.nodes()).index(node) for node in connected])

    plt.figure(figsize=(8, 8))

    k = layout_k_factor / math.sqrt(len(G.nodes()))
    layout = nx.spring_layout(G, k=k, iterations=layout_iteration)
    layout_thesis = {}
    for i, pos in enumerate(layout.values()):
        layout_thesis[i] = pos

    pr = nx.pagerank(G)
    pr_values = np.array([pr[node] for node in G.nodes()])
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['others'], node_color="#ffffff", alpha=0.7, node_size=pr_values[nodes_thesis['others']] * node_size_factor, edgecolors="#888888")
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['A7'], node_color=color_thesis['A7'], alpha=0.6, node_size=pr_values[nodes_thesis['A7']] * node_size_factor * 0.5, linewidths=2.0, edgecolors=color_thesis['A7'])
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['A7'], node_color=color_thesis['A7'], alpha=0.4, node_size=pr_values[nodes_thesis['A7']] * node_size_factor)
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['A8'], node_color=color_thesis['A8'], alpha=0.4, node_size=pr_values[nodes_thesis['A8']] * node_size_factor * 0.7, linewidths=1.0, edgecolors=color_thesis['A8'])
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['A8'], node_color=color_thesis['A8'], alpha=0.3, node_size=pr_values[nodes_thesis['A8']] * node_size_factor)
    nx.draw_networkx_nodes(G, layout_thesis, nodelist=nodes_thesis['A9'], node_color=color_thesis['A9'], alpha=0.8, node_size=pr_values[nodes_thesis['A9']] * node_size_factor)

    edge_width = [weight * edge_width_factor for _, _, weight in G.edges(data='weight')]
    nx.draw_networkx_edges(G, layout, alpha=0.4, edge_color="darkgrey", width=edge_width)

    nx.draw_networkx_labels(G, layout, font_family='Hiragino Sans', font_size=10)

    plt.axis('off')
    plt.savefig(f'{result_dir}/{filename}.pdf', dpi=300)
    plt.show()