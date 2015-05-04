__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree
from util import ml_util
import numpy
import random
import re

SENTENCES = 'sentences'
PARSE_TREE = 'parsetree'
TXT = 'text'
TUPLES = 'tuples'


def getSuccessFailure():
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(Novels)
    success_files, failure_files = {}, {}
    for x in core_nlp_files_dict.keys():
        W, L = [], []
        for y in core_nlp_files_dict[x]:
            if re.search('success',y[0]):
                W.append(y)
            else:
                L.append(y)
        if len(W) > 0:
            success_files[x] = W
        if len(L) > 0:
            failure_files[x] = L
    return (success_files, failure_files)


def getTree(core_nlp_files_dict, genre):
    core_nlp_files = core_nlp_files_dict[genre]
    productions = {}
    for files in core_nlp_files:
        dictionary = dict()
        product = {}
        with open(files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 1
            line = lines[0]
            line = 'dictionary=' + line
            exec(line)
            sentences = dictionary[SENTENCES]
            for sent in sentences:
                parsetree = sent[PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                product[parsetree] = t
        productions[files[0]] = product
    return productions


def doVarianceMeasure(core_nlp_files_dict, genre):
    productions = getTree(core_nlp_files_dict, genre)
    for files in productions.keys():
        product = productions[files]
        rules = []
        diff_productions = {}
        for sents in product.keys():
            t = product[sents]
            for prod in t.productions():
                if prod not in diff_productions:
                    diff_productions[prod] = 0.0
                diff_productions[prod] += 1.0
        rules.append(len(diff_productions))
    return (numpy.mean(rules), numpy.std(rules), numpy.var(rules))

def doComplexCompound(core_nlp_files_dict, l, genre):
    productions = getTree(core_nlp_files_dict, genre)
    featureV =[]
    label = []
    for files in productions.keys():
        product = productions[files]
        types = {'SIMPLE':0, 'COMPLEX':0, 'COMPOUND':0, 'COMPLEX-COMPOUND':0, 'OTHER':0}
        for sents in product.keys():
            t = product[sents]
            types[checkProduction(t)] += 1
        fv = [types['SIMPLE'], types['COMPLEX'], types['COMPOUND'], types['COMPLEX-COMPOUND']]
        featureV.append(fv)
        label.append(l)
    return featureV, label


def checkProduction(tree):
    topLevel = str(tree.productions()[1])
    tags = tree.pos()
    tags = [x[1] for x in tags]
    SBAR = True if 'SBAR' in tags else False
    if re.search('. S .', topLevel.split('>')[1]):
        if not SBAR:
            return 'COMPOUND'
        else:
            return 'COMPLEX-COMPOUND'
    else:
        if re.search('. VP .', topLevel.split('>')[1]):
            if not SBAR:
                return 'SIMPLE'
            else:
                return 'COMPLEX'
    return 'OTHER'


def shuffle(data, labels):
    start = len(data) - (6*len(data)/10)
    end = len(data) - (5*len(data)/10)
    x = [i for i in range(len(data))]
    random.shuffle(x)
    traindata = [data[i] for i in x[:start]]
    traindata.extend([data[i] for i in x[end:]])
    trainlabels = [labels[i] for i in x[:start]]
    trainlabels.extend([labels[i] for i in x[end:]])
    testdata = [data[i] for i in x[start:end]]
    testlabels = [labels[i] for i in x[start:end]]
    return (traindata,trainlabels,testdata, testlabels)

Novels = '/home/sriganesh/Documents/NLP/NLP_Project/core_nlp'
#doClassification()
meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
x, y = getSuccessFailure()
for genre in x:
    if genre == 'Science Fiction' or genre == 'Short Stories':
        continue
    fv1, lb1 = doComplexCompound(x,'S',genre)
    fv2, lb2 = doComplexCompound(y,'F',genre)
    fv1.extend(fv2)
    lb1.extend(lb2)
    ml_util.splitTrainAndTestData(meta_dict[genre], )
    print genre, "Accuracy:", accuracy_score(label, mylabel)
#p1 = doVarianceMeasure(x)
#p2 = doVarianceMeasure(y)
#for x in p1.keys():
#    print x
#    print "Success:", p1[x]
#    print "Failure:", p2[x]