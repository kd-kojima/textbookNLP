# -*- coding: utf-8 -*-
# co-occurence

import collections
import itertools
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

import MeCab
import neologdn
import unicodedata

class Combinations():
    def __init__(self, nouns):
        self.combs = nouns.make_comb()
        self.jaccard_table = None

    def print(self):
        print(self.combs)

    def jaccard(self):
        count_comb = collections.Counter(self.combs)
        table_comb = pd.DataFrame(
            [[key[0], key[1], value] for key, value in count_comb.items()],
            columns=['word1', 'word2', 'inter']
        )

        words = []
        for c in self.combs:
            words.extend(c)
        count_word = collections.Counter(words)
        table_word = pd.DataFrame(
            [[key, value] for key, value in count_word.items()],
            columns=['word', 'count']
        )

        table = pd.merge(
            table_comb, table_word.rename(columns={'word': 'word1'}), on='word1', how='left'
        ).rename(columns={'count': 'count1'}).merge(
            table_word.rename(columns={'word': 'word2'}), on='word2', how='left'
        ).rename(columns={'count': 'count2'}).assign(
            union = lambda x: x.count1 + x.count2 - x.inter
        ).assign(
            jaccard = lambda x: x.inter / x.union
        ).sort_values(
            ['jaccard', 'inter'], ascending=[False, False]
        )

        self.jaccard_table = table
        return table

    def print_jaccard_table(self):
        print('\n--- jaccard index ---')
        print(self.jaccard_table)
        print('---------------------')

    def jaccard_freq_table(self):
        if self.jaccard_table is None:
            self.jaccard()

        jaccard_index = self.jaccard_table['jaccard']
        bins = [0., 0.01, 0.02, 0.04, 1.]
        freq = jaccard_index.value_counts(bins=bins, sort=False)
        rel_freq = freq / jaccard_index.count()
        rel_cum_freq = rel_freq.cumsum()
        dist = pd.DataFrame(
            {
                'freq': freq,
                'rel_freq': rel_freq,
                'rel_cum_freq': rel_cum_freq
            }, index = freq.index
        )
        print('\n--- freq table ---')
        print(dist)
        print('------------------')

    def plot_network(
        self,
        n_word_lower = 10,
        edge_threshold = 0.,
        figsize = (15, 15),
        fontfamily = 'Yu Gothic',
        fontsize = 14,
        restitution_coef = 0.15,
        filepath = None
    ):
        if self.jaccard_table is None:
            self.jaccard()

        self.print_jaccard_table()
        self.jaccard_freq_table()

        # n_word_lower = int(input('Input lower limit of number of words'))

        # edge_threshold = input('Input edge threshold. (default: 0.)')
        # edge_threshold = float(edge_threshold) if edge_threshold != '' else 0.

        # restitution_coef = input('Input restitution coefficient. (default: 0.15)')
        # restitution_coef = float(restitution_coef) if restitution_coef != '' else 0.15

        # filepath = input('Input save file path if you want to save plot.')

        data = self.jaccard_table.query(
            f'count1 >= {n_word_lower} and count2 >= {n_word_lower}'
        ).rename(
            columns={'word1': 'node1', 'word2': 'node2', 'jaccard': 'weight'}
        )

        nodes = list(set(data['node1'].tolist() + data['node2'].tolist()))
        
        plt.figure(figsize=figsize)

        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i in range(len(data)):
            row = data.iloc[i]
            if row['weight'] >= edge_threshold:
                G.add_edge(row['node1'], row['node2'], weight=row['weight'])

        isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]
        for n in isolated:
            G.remove_node(n)

        pos = nx.spring_layout(G, k=restitution_coef)

        pr = nx.pagerank(G)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=list(pr.values()),
            cmap=plt.cm.Reds,
            alpha=0.7,
            node_size=[60000*v for v in pr.values()]
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=fontsize,
            font_family=fontfamily,
            font_weight='bold'
        )

        edge_width = [d['weight'] * 100 for (u, v, d) in G.edges(data=True)]
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.4,
            edge_color='darkgrey',
            width=edge_width
        )

        plt.axis('off')
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300)

        plt.show()