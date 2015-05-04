__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree
from util import ml_util
from util import utils
import numpy
import re

SENTENCES = 'sentences'
PARSE_TREE = 'parsetree'
TXT = 'text'
TUPLES = 'tuples'


def getSuccessFailure():
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(\
        NovelMetaGenerator.CORE_NLP_BASE,\
        NovelMetaGenerator.SYNSET_WSD_TAG_PATTERN)
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
        key = files[0].replace(NovelMetaGenerator.CORE_NLP_FILE_SUFFIX, '')
        productions[key] = product
    return productions


def doVarianceMeasure(core_nlp_files_dict, genre):
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    meta_dict_for_genre = meta_dict[genre]
    productions = getTree(core_nlp_files_dict, genre)
    for files in meta_dict_for_genre.keys():
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
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    meta_dict_for_genre = meta_dict[genre]
    productions = getTree(core_nlp_files_dict, genre)
    featureV =[]
    label = []
    for files in meta_dict_for_genre.keys():
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



def extractHeightDistributionFeature(core_nlp_files):
    max_ht = 0
    height_distribution_feature = dict()
    for genre_file_path, genre_file_name in core_nlp_files:
        dictionary = dict()
        with open(genre_file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            line = lines[0]
            line = 'dictionary=' + line
            exec(line)
            # print genre_file_path, dictionary
            sentences = dictionary[NovelMetaGenerator.SENTENCES]
            height_feature_for_file = dict()
            for sent in sentences:
                parsetree = sent[NovelMetaGenerator.PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                height = t.height()
                if height > max_ht:
                    max_ht = height
                if height not in height_feature_for_file:
                    height_feature_for_file[height] = 0.0
                height_feature_for_file[height] += 1.0

            key = genre_file_name.replace(NovelMetaGenerator.CORE_NLP_FILE_SUFFIX, '')
            height_distribution_feature[key] = height_feature_for_file
    height_distribution_feature = utils.normalize_dist(height_distribution_feature, set([i for i in range(1, max_ht)]))
    return height_distribution_feature


def doClassification():
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    # x, y = getSuccessFailure()
    # for genre in x:
    #     if genre == 'Science Fiction' or genre == 'Short Stories':
    #         continue
    #     fv1, lb1 = doComplexCompound(x,'S',genre)
    #     fv2, lb2 = doComplexCompound(y,'F',genre)
    #     fv1.extend(fv2)
    #     lb1.extend(lb2)
    for genre in meta_dict.keys():
        if genre == 'Science Fiction' or genre == 'Short Stories':
            continue
        meta_dict_for_genre = meta_dict[genre]
        core_nlp_files = core_nlp_files_dict[genre]
        feature_dict = extractHeightDistributionFeature(core_nlp_files)
        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict)
        accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print genre, ':', accuracy

doClassification()
