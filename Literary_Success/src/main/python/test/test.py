__author__ = 'santhosh'


from feature_wise_predictor import POSFeatureUtil
from feature_wise_predictor import SyntaticTreeFeaturesUtil
from feature_wise_predictor import ConnotationFeatureUtil

print POSFeatureUtil.doClassification()
SyntaticTreeFeaturesUtil.doClassification()
ConnotationFeatureUtil.doClassification()
