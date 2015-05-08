__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import data_reader


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

def normalize_dist(feature_dict, diff_pos):
    for f in feature_dict:
        feature_dict_for_file = feature_dict[f]
        sum_of_production_rules = sum(feature_dict_for_file.values())
        feature_dict[f] = {k:(feature_dict_for_file[k]/sum_of_production_rules) if k in feature_dict_for_file else 0.0\
                                        for k in diff_pos}
    return feature_dict

def extractPOSFeaturesFromCoreNLPFiles(core_nlp_files):
    diff_pos = set()
    feature_dict = dict()
    for core_nlp_file in core_nlp_files:
        genre_file_path, genre_file_name = core_nlp_file
        curr_file_feature = dict()
        trees = data_reader.readCoreNLPFileAndReturnTree(core_nlp_file)
        for t in trees:
            wordAndtags = t.pos()
            for word, pos in wordAndtags:
                if pos not in curr_file_feature:
                    curr_file_feature[pos] = 0.0
                curr_file_feature[pos] += 1.0
                diff_pos.add(pos)
        key = genre_file_name.replace(NovelMetaGenerator.CORE_NLP_FILE_SUFFIX, '')
        feature_dict[key] = curr_file_feature
    feature_dict = normalize_dist(feature_dict, diff_pos)
    return feature_dict