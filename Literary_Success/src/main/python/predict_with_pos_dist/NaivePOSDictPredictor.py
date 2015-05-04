__author__ = 'santhosh'

content = 'meta_dict='
import numpy
import random
from sklearn.cluster import k_means
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from util import ml_util


CLASS = 'class'
TAGS = 'TAGS'
SUCCESS_PATTERN = 'SUCCESS'
FAILURE_PATTERN = 'FAILURE'

def readMetaInfo():
    content = 'meta_dict='
    with open('../../../../novel_meta_pos_bigram.meta', 'r') as f:
        meta_dict = dict()
        content = content + f.readline()
        exec (content)
        return meta_dict


def doClassification():
    meta_dict = readMetaInfo()
    feature_dict = dict()
    accuracy_for_genre = dict()
    for genre in meta_dict.keys():
        meta_dict_for_genre = meta_dict[genre]
        file_names = [file_name for file_name in meta_dict_for_genre]
        feature_dict = {file_name: dict() for file_name in file_names}
        diff_pos = list(set([pos_tag for file_name in file_names for pos_tag in meta_dict_for_genre[file_name][TAGS]]))
        for file_name in file_names:
            for pos_tag in diff_pos:
                if pos_tag not in meta_dict_for_genre[file_name][TAGS]:
                    meta_dict_for_genre[file_name][TAGS][pos_tag] = 0.0
                feature_dict[file_name][pos_tag] = meta_dict_for_genre[file_name][TAGS][pos_tag]

        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict)

        accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        accuracy_for_genre[genre] = accuracy
    return accuracy_for_genre
