__author__ = 'santhosh'

from pattern.en import parsetree
from pattern.en import tag


for word, pos in tag('I feel *happy*!'):
    print word, pos
s = parsetree('The cat sat on the mat.', relations=True, lemmata=True)
print repr(s)
from pattern.en import ngrams, sentiment
print ngrams("I am eating pizza.", n=2) # bigrams

print sentiment('Wonderfully awful! :-)').assessments