__author__ = 'santhosh'

from pattern.en import parsetree
from pattern.en import tag


# for word, pos in tag('I feel *happy*!'):
#     print word, pos
# s = parsetree('The cat sat on the mat.', relations=True, lemmata=True)
# print repr(s)

# from pattern.en import parse
# s = 'This is my sample'
# s = parse(s, relations=True, lemmata=True)
# print s



from nltk.corpus import wordnet as wn



def getSenseSimilarity(worda,wordb):
	wordasynsets = wn.synsets(worda)
	wordbsynsets = wn.synsets(wordb)

	synsetnamea = [wn.synset(str(syns.name)) for syns in wordasynsets]

	synsetnameb = [wn.synset(str(syns.name)) for syns in wordbsynsets]

	for sseta, ssetb in [(sseta,ssetb) for sseta in synsetnamea\

	for ssetb in synsetnameb]:

		pathsim = sseta.path_similarity(ssetb)

		wupsim = sseta.wup_similarity(ssetb)

		if pathsim != None:
			print "Path Sim Score: ",pathsim," WUP Sim Score: ",wupsim,\
			"\t",sseta.definition, "\t", ssetb.definition



getSenseSimilarity('cricket','score')
