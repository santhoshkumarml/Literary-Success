__author__ = 'santhosh'

from pattern.en import parsetree
from pattern.en import tag
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')


# for word, pos in tag('I feel *happy*!'):
#     print word, pos
# s = parsetree('The cat sat on the mat.', relations=True, lemmata=True)
# print repr(s)

# from pattern.en import parse
# s = 'This is my sample'
# s = parse(s, relations=True, lemmata=True)
# print s

from pywsd import lesk as lsk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

data = lsk.adapted_lesk(u'I killed Cricket', u'Cricket')
ranked_synsets = data
probs = 0.0
for ranked_synset in ranked_synsets:
    prob, syn = ranked_synset
    print prob, syn.name()
    probs += prob
print probs

