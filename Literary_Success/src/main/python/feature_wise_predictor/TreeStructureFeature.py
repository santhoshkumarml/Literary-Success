__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree
from util import ml_util
from feature_wise_predictor import SyntaticTreeFeaturesUtil
from util import utils
import numpy
import re

HEIGHT = 'HEIGHT'
WIDTH = 'WIDTH'
COMPLEX_COMPOUND_FEATURE = 'COMPLEX_COMPOUND'
LOOSE_PERIODIC_FEATURE = 'LOOSE_PERIODIC'
TREE_IMBALANCE = 'TREE_IMBALANCE'
ALL_FEATURES = set([HEIGHT, WIDTH, COMPLEX_COMPOUND_FEATURE, LOOSE_PERIODIC_FEATURE, TREE_IMBALANCE])


SIMPLE = 'SIMPLE'
COMPOUND = 'COMPOUND'
COMPLEX = 'COMPLEX'
COMPLEX_COMPOUND = 'COMPLEX-COMPOUND'
COMPLEX_COMPOUND_OTHER = 'COMPLEX_COMPOUND_OTHER'

PERIODIC = 'PERIODIC'
LOOSE = 'LOOSE'
LOOSE_PERIODIC_OTHER = 'LOOSE_PERIODIC_OTHER'

DIFF_CPX_TYPES = set([SIMPLE, COMPLEX, COMPOUND,  COMPLEX_COMPOUND, COMPLEX_COMPOUND_OTHER])
DIFF_LP_TYPES = set([LOOSE, PERIODIC, LOOSE_PERIODIC_OTHER])

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
            sentences = dictionary[SyntaticTreeFeaturesUtil.SENTENCES]
            for sent in sentences:
                parsetree = sent[SyntaticTreeFeaturesUtil.PARSE_TREE]
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


def checkLoosePeriodic(tree):
    k = 1
    Ltop = len(str(tree.productions()[1]).split('>')[1].strip().strip('.').strip(',').split())
    while k <= Ltop:
        topLevel = str(tree.productions()[k]).split('>')[1]
        VP = True if re.search('. VP .', topLevel) else False
        SSBAR = True if re.search('. S .', topLevel) or re.search('. SBAR .', topLevel) else False
        if not VP:
            if SSBAR:
                return PERIODIC
        else:
            if SSBAR:
                return LOOSE
        k += 1
    return LOOSE_PERIODIC_OTHER


def checkComplexCompound(tree):
    topLevel = str(tree.productions()[1])
    tags = tree.pos()
    tags = [x[1] for x in tags]
    SBAR = True if 'SBAR' in tags else False
    if re.search('. S .', topLevel.split('>')[1]):
        if not SBAR:
            return COMPOUND
        else:
            return COMPLEX_COMPOUND
    else:
        if re.search('. VP .', topLevel.split('>')[1]):
            if not SBAR:
                return SIMPLE
            else:
                return COMPLEX
    return COMPLEX_COMPOUND_OTHER



def extractDeepSyntaticFeature(core_nlp_files, features = None):
    if not features:
        features = ALL_FEATURES
    max_ht = 0
    deep_syntactic_feature_dict = dict()
    for genre_file_path, genre_file_name in core_nlp_files:
        dictionary = dict()
        with open(genre_file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            line = lines[0]
            line = 'dictionary=' + line
            exec(line)

            key = genre_file_name.replace(NovelMetaGenerator.CORE_NLP_FILE_SUFFIX, '')
            deep_syntactic_feature_dict[key] = dict()

            sentences = dictionary[NovelMetaGenerator.SENTENCES]
            height_feature_for_file = dict()
            complex_compound_feature_for_file = dict()
            loose_periodic_feature_for_file = dict()
            for sent in sentences:
                parsetree = sent[NovelMetaGenerator.PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                if HEIGHT in features:
                    height = t.height()
                    if height > max_ht:
                        max_ht = height
                    if height not in height_feature_for_file:
                        height_feature_for_file[height] = 0.0
                    height_feature_for_file[height] += 1.0

                if COMPLEX_COMPOUND_FEATURE in features:
                    compex_compond_type = checkComplexCompound(t)
                    if compex_compond_type == COMPLEX_COMPOUND_OTHER:
                        continue
                    if compex_compond_type not in complex_compound_feature_for_file:
                        complex_compound_feature_for_file[compex_compond_type] = 0.0
                    complex_compound_feature_for_file[compex_compond_type] += 1.0

                if LOOSE_PERIODIC_FEATURE in features:
                    loose_periodic_type = checkLoosePeriodic(t)
                    if loose_periodic_type == LOOSE_PERIODIC_OTHER:
                        continue
                    if loose_periodic_type not in loose_periodic_feature_for_file:
                        loose_periodic_feature_for_file[loose_periodic_type] = 0.0
                    loose_periodic_feature_for_file[loose_periodic_type] += 1.0

            if HEIGHT in features:
                deep_syntactic_feature_dict[key][HEIGHT] = height_feature_for_file

            if COMPLEX_COMPOUND_FEATURE in features:
                deep_syntactic_feature_dict[key][COMPLEX_COMPOUND_FEATURE] = complex_compound_feature_for_file

            if LOOSE_PERIODIC_FEATURE in features:
                deep_syntactic_feature_dict[key][LOOSE_PERIODIC_FEATURE] = loose_periodic_feature_for_file

        #Normalize and Induce Feature
    for f in deep_syntactic_feature_dict.keys():
        if HEIGHT in features:
            deep_syntactic_feature_dict[f][HEIGHT] =\
                utils.normalize_dist(deep_syntactic_feature_dict[f][HEIGHT], [i for i in range(1, max_ht)])

            for k in deep_syntactic_feature_dict[f][HEIGHT].keys():
                deep_syntactic_feature_dict[f][HEIGHT+str(k)] = deep_syntactic_feature_dict[f][HEIGHT][k]

            del deep_syntactic_feature_dict[f][HEIGHT]

        if COMPLEX_COMPOUND_FEATURE in features:
            deep_syntactic_feature_dict[f][COMPLEX_COMPOUND_FEATURE] =\
                utils.normalize_dist(deep_syntactic_feature_dict[f][COMPLEX_COMPOUND_FEATURE], DIFF_CPX_TYPES)

            for k in deep_syntactic_feature_dict[f][COMPLEX_COMPOUND_FEATURE].keys():
                deep_syntactic_feature_dict[f][k] = deep_syntactic_feature_dict[f][COMPLEX_COMPOUND_FEATURE][k]

            del deep_syntactic_feature_dict[f][COMPLEX_COMPOUND_FEATURE]

        if LOOSE_PERIODIC_FEATURE in features:
            deep_syntactic_feature_dict[f][LOOSE_PERIODIC_FEATURE] =\
                utils.normalize_dist(deep_syntactic_feature_dict[f][LOOSE_PERIODIC_FEATURE], DIFF_LP_TYPES)

            for k in deep_syntactic_feature_dict[f][LOOSE_PERIODIC_FEATURE].keys():
                deep_syntactic_feature_dict[f][k] = deep_syntactic_feature_dict[f][LOOSE_PERIODIC_FEATURE][k]

            del deep_syntactic_feature_dict[f][LOOSE_PERIODIC_FEATURE]

    return deep_syntactic_feature_dict


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
        feature_dict = extractDeepSyntaticFeature(core_nlp_files, features=set([LOOSE_PERIODIC_FEATURE]))
        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict)
        accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print genre, ':', accuracy
