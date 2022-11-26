import json
import MeCab
import prep, cooc

args = json.load(open('./args.json', 'r'))

stopwords = []
with open(args["stopwords"], 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')

R = prep.RawText(filepath=f'{args["rawdir"]}/{args["filename"]}.txt')
R.normalize()
R.count_words(mecab=MeCab.Tagger(f'-d \"{args["ipadic"]}\" -u \"{args["userdic"]}\" -Owakati'))
# R.print()

S = prep.Sentences(R, stopwords=stopwords)
S.save(f'{args["textdir"]}/{args["filename"]}.txt')
# S.print()
N = prep.NounSentences(S, mecab=MeCab.Tagger(f'-d \"{args["ipadic"]}\" -u \"{args["userdic"]}\" -Odump'))
N.save(f'{args["noundir"]}/{args["filename"]}.txt')
# N.print()

C = cooc.Combinations(N)
C.jaccard()

C.plot_network(
    n_word_lower = args["n_word_lower"],
    edge_threshold = args["edge_threshold"],
    restitution_coef = args["restitution_coef"],
    filepath = f'{args["resultdir"]}/{args["n_word_lower"]}_{args["edge_threshold"]}_{args["restitution_coef"]}_{args["filename"]}.png'
)