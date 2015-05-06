__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree
import numpy
import random
from sklearn.linear_model import LogisticRegression
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SENTENCES = 'sentences'
PARSE_TREE = 'parsetree'
TXT = 'text'
TUPLES = 'tuples'

def getConsituentTreeDistribution(core_nlp_files):
    diff_productions = dict()
    production_dict_for_files = dict()
    for genre_file_path, genre_file_name in core_nlp_files:
        production_dict = dict()
        dictionary = dict()
        with open(genre_file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            line = lines[0]
            line = 'dictionary=' + line
            exec(line)
            # print genre_file_path, dictionary
            sentences = dictionary[SENTENCES]
            for sent in sentences:
                parsetree = sent[PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                prods = t.productions()
                for prod in prods:
                    if prod not in diff_productions:
                        diff_productions[prod] = 0.0
                    if prod not in production_dict:
                        production_dict[prod] = 0.0
                    diff_productions[prod] += 1.0
                    production_dict[prod] += 1.0
            production_dict_for_files[genre_file_name.replace('_corenlp1000.txt', '.txt')] = production_dict
    return production_dict_for_files, diff_productions


def normalize_dist(production_dict_for_files, diff_productions):
    for f in production_dict_for_files:
        prod_dict_for_file = production_dict_for_files[f]
        sum_of_production_rules = sum(prod_dict_for_file.values())
        production_dict_for_files[f] = {k:(prod_dict_for_file[k]/sum_of_production_rules) if k in prod_dict_for_file else 0.0\
                                        for k in diff_productions.keys()}
    return production_dict_for_files


def splitTrainAndTestData(meta_dict_for_genre, production_dict_for_files, split = 0.7):
    file_names = [file_name for file_name in meta_dict_for_genre]


    n_samples = len(file_names)
    n_features = len(production_dict_for_files[file_name].values())
    data = numpy.zeros(shape=(n_samples, n_features))


    class_wise_genre_file = {NovelMetaGenerator.SUCCESS_PATTERN:[],NovelMetaGenerator.FAILURE_PATTERN:[]}

    for file_name in meta_dict_for_genre:
        if meta_dict_for_genre[file_name][NovelMetaGenerator.CLASS] == NovelMetaGenerator.SUCCESS_PATTERN:
            class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN].append(file_name)
        else:
            class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN].append(file_name)
    total_success_files = len(class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN])
    total_failure_files = len(class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN])
    success_train_size, failure_train_size = int(total_success_files*split), int(total_failure_files*split)

    random_train_success_idx = set(random.sample(xrange(total_success_files), success_train_size))

    random_train_failure_idx = set(random.sample(xrange(total_failure_files), failure_train_size))

    train_data = ([],[])
    test_data = ([],[])

    for i in range(total_success_files):
            file_name = class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN][i]
            if i in random_train_success_idx:
                train_data[0].append(list(production_dict_for_files[file_name].values()))
                train_data[1].append(1)
            else:
                test_data[0].append(list(production_dict_for_files[file_name].values()))
                test_data[1].append(1)

    for i in range(total_failure_files):
        file_name = class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN][i]
        if i in random_train_failure_idx:
            train_data[0].append(list(production_dict_for_files[file_name].values()))
            train_data[1].append(0)
        else:
            test_data[0].append(list(production_dict_for_files[file_name].values()))
            test_data[1].append(0)

    return train_data, test_data




def doClassification():
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE)
    novel_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE)
    for genre in core_nlp_files_dict:
        if genre == 'Science Fiction' or genre == 'Short Stories':
            continue
        meta_dict_for_genre = meta_dict[genre]
        core_nlp_files = core_nlp_files_dict[genre]
        production_dict_for_files, diff_productions = getConsituentTreeDistribution(core_nlp_files)
        production_dict_for_files = normalize_dist(production_dict_for_files, diff_productions)
        train_data, test_data = splitTrainAndTestData(meta_dict_for_genre, production_dict_for_files)
        log_r = LogisticRegression()
        train_data, train_result = train_data
        test_data, test_result = test_data
        log_r.fit(train_data, train_result)
        accuracy = 0.0
        for i in range(len(test_data)):
            label = int(log_r.predict(test_data[i]))
            if label == test_result[i]:
                accuracy += 1.0
        accuracy = accuracy/len(test_data)
        print genre, ':', accuracy


def getSuccessFailure():
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(Novels, NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
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
            lines = f.readlines()[:100]
            for line in lines:
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

def traverseLabels(tree):
    if not isinstance(tree, ParentedTree):
        return []
    output = [tree.label()]
    if len(tree) > 0:
        for child in tree:
            output = output + traverseLabels(child)
    return output

def checkVoice1(tree):
    t = tree
    st = []

    found = False
    while isinstance(t, ParentedTree) and t.label() != 'S':
        children = [child for child in t]
        st.extend(children)
        t = st.pop()
        found = True

    if found:
        children = [child for child in t]
        k = 0
        while k < len(children):
            result = traverseLabels(children[k])
            if isinstance(children[k], ParentedTree) and children[k].label() != 'VP':
                if 'S' in result or 'SBAR' in result:
                    return 'PERIODIC'
            else:
                if 'S' in result or 'SBAR' in result:
                    return 'LOOSE'
            k += 1

    return 'OTHER'



def checkProduction(tree):
    topLevel = str(tree.productions()[1])
    tags = traverseLabels(tree)
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


def doLoosePeriodic(core_nlp_files_dict, l, genre):
    productions = getTree(core_nlp_files_dict, genre)
    featureV = []
    label = []
    for files in productions.keys():
        product = productions[files]
        types = {'LOOSE':0, 'PERIODIC':0, 'OTHER':0}
        for sents in product.keys():
            t = product[sents]
            types[checkVoice1(t)] += 1
        fv = [types['LOOSE'], types['PERIODIC']]
        featureV.append(fv)
        label.append(l)
    return featureV, label

def checkVoice(tree):
    k = 1
    #print "Productions:",tree.productions()
    #print "TopLevel:", str(tree.productions()[1]).split('>')[1]
    Ltop = len(str(tree.productions()[1]).split('>')[1].strip().strip('.').strip(',').split())
    while k <= Ltop:
        topLevel = str(tree.productions()[k]).split('>')[1]
        #print k,"Level Tags:", topLevel
        VP = True if re.search('. VP .', topLevel) else False
        SSBAR = True if re.search('. S .', topLevel) or re.search('. SBAR .', topLevel) else False
        #print "VP:", VP, "SSBAR:", SSBAR
        if not VP:
            if SSBAR:
                return 'PERIODIC'
        else:
            if SSBAR:
                return 'LOOSE'
        k += 1
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

Novels = NovelMetaGenerator.CORE_NLP_BASE
#doClassification()
x, y = getSuccessFailure()
for genre in x:
    if genre != 'Adventure Stories':
        continue
    fv1, lb1 = doComplexCompound(x,'S',genre)
    fv2, lb2 = doComplexCompound(y,'F',genre)
    print fv1
    print fv2
    fv1.extend(fv2)
    lb1.extend(lb2)
    train, tlabel, test, label = shuffle(fv1, lb1)
    for j in range(len(train)):
        print train[j], tlabel[j]
    clf = LinearSVC(C=0.5)
    clf.fit(train, tlabel)
    mylabel = []
    for k in test:
        mylabel.append(clf.predict(k))
    print genre, "Accuracy:", accuracy_score(label, mylabel)
#p1 = doVarianceMeasure(x)
#p2 = doVarianceMeasure(y)
#for x in p1.keys():
#    print x
#    print "Success:", p1[x]
#    print "Failure:", p2[x]

