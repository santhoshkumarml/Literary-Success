__author__ = 'santhosh'


from feature_wise_predictor import POSFeatureUtil
from feature_wise_predictor import SyntaticTreeFeaturesUtil
from feature_wise_predictor import ConnotationFeatureUtil
from feature_wise_predictor import TreeStructureFeature

# POSFeatureUtil.doClassification()
# SyntaticTreeFeaturesUtil.doClassification()
# ConnotationFeatureUtil.doClassification()
TreeStructureFeature.doClassification()

# from util import NovelMetaGenerator
# NovelMetaGenerator.extractMetaDataAndTagCoreNLP()

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

