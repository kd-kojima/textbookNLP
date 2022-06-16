import json
import prep, cooc

args = json.load(open('./args.json', 'r'))

R = prep.RawText(filepath=f'{args["rawdir"]}/{args["rawfile"]}')
R.normalize()
R.count_words()
# R.print()

S = prep.Sentences(R)
# S.print()
N = prep.NounSentences(S)
# N.print()

C = cooc.Combinations(N)
C.jaccard()

C.plot_network(
    n_word_lower = args["n_word_lower"],
    edge_threshold = args["edge_threshold"],
    restitution_coef = args["restitution_coef"],
    filepath = f'{args["resultdir"]}/{args["resultfile"]}'
)