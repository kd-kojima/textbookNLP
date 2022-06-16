import json
import prep, cooc

paths = json.load(open('./path.json', 'r'))

R = prep.RawText(filepath=f'{paths["rawdir"]}/{paths["rawfile"]}')
R.normalize()
R.count_words()
# R.print()

S = prep.Sentences(R)
# S.print()
N = prep.NounSentences(S)
N.print()
N.make_comb()
