__author__ = 'santhosh'

from pattern.en import parsetree
from pattern.en import tag


# for word, pos in tag('I feel *happy*!'):
#     print word, pos
# s = parsetree('The cat sat on the mat.', relations=True, lemmata=True)
# print repr(s)

from pattern.en import parse
s = 'This is my sample'
s = parse(s, relations=True, lemmata=True)
print s