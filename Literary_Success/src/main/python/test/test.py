__author__ = 'santhosh'


from feature_wise_predictor import POSFeatureUtil
from feature_wise_predictor import SyntaticTreeFeaturesUtil
from feature_wise_predictor import ConnotationFeatureUtil
from feature_wise_predictor import TreeStructureFeature
from util import NovelMetaGenerator
from util import ml_util

# POSFeatureUtil.doClassification()
# SyntaticTreeFeaturesUtil.doClassification()
# ConnotationFeatureUtil.doClassification()
# TreeStructureFeature.doClassification()

from util import NovelMetaGenerator
NovelMetaGenerator.extractMetaDataAndTagCoreNLP(genres=set(['Love Stories']))

# import nltk
# import jsonrpclib
# from simplejson import loads
#
# nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')
#
# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#
# text = '''Punkt knows that the periods in Mr. Smith and Johann S. Bach
# do not mark sentence boundaries.  And sometimes sentences
# can start with non-capitalized words.  i is a good variable
# name.'''
# sentences = sent_detector.tokenize(text.strip())
# server = jsonrpclib.Server("http://localhost:8080")
# for string in sentences:
#     result = loads(server.parse(string))
#     sents = result['sentences']
#     for sent in sents:
#         print sent['text']

# from util import NovelMetaGenerator
# NovelMetaGenerator.extractSysetDistributionForWORDS()
#
# def doClassification():
#     meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
#     core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
#                                                                     NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
#     for genre in meta_dict.keys():
#         if genre != 'Adventure Stories' and genre != 'Love Stories':
#             continue
#         meta_dict_for_genre = meta_dict[genre]
#         core_nlp_files = core_nlp_files_dict[genre]
#
#         feature_dict1 = SyntaticTreeFeaturesUtil.extractSyntacticFeatures(core_nlp_files)
#         feature_dict2 = TreeStructureFeature.extractDeepSyntaticFeature(core_nlp_files)
#
#         for f in feature_dict1.keys():
#             for k in feature_dict2[f].keys():
#                 feature_dict1[f][k] = feature_dict2[f][k]
#
#         train_data, train_result, test_data, test_result =\
#             ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict1)
#         accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
#         print genre, ':', accuracy
#
#
# doClassification()
#
# from nltk.tree import ParentedTree
# f1 = '/media/santhosh/Data/workspace/nlp_project/core_nlp1/Adventure_Stories/as_fold1/success1/1965_corenlp1000.txt'
# dictionary = dict()
# height_feature_for_file = dict()
# horizontal_imbalance_feature_for_file = dict()
# vertical_imbalance_feature_for_file = dict()
#
# with open(f1) as f:
#     lines = f.readlines()
#     lines = lines[:100]
#     sent_count = 0
#     for line in lines:
#         line = 'dictionary='+line
#         exec(line)
#         sent_count += 1
#         sentences = dictionary[NovelMetaGenerator.SENTENCES]
#         for sent in sentences:
#             parsetree = sent[NovelMetaGenerator.PARSE_TREE]
#             txt = sent[NovelMetaGenerator.TXT]
#             t = ParentedTree.fromstring(parsetree)
#             height = t.height()
#             if height not in height_feature_for_file:
#                 height_feature_for_file[height] = 0.0
#             height_feature_for_file[height] += 1.0
#             furcation_node_dict = TreeStructureFeature.getFurcationNodesAndHeight(t)
#             horizontal_imbalance_sent = TreeStructureFeature.horizontal_imbalance(furcation_node_dict)
#             horizontal_imbalance_feature_for_file[TreeStructureFeature.HORIZONTAL_TREE_IMBALANCE+str(sent_count)] =\
#                 horizontal_imbalance_sent
#
#             vertical_imbalance_sent = TreeStructureFeature.vertical_imbalance(furcation_node_dict)
#             vertical_imbalance_feature_for_file[TreeStructureFeature.VERTICAL_TREE_IMBALANCE+str(sent_count)] =\
#                 vertical_imbalance_sent
#     print height_feature_for_file
#     print horizontal_imbalance_feature_for_file
#     print vertical_imbalance_feature_for_file
# ConnotationFeatureUtil.doClassification()

