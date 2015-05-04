__author__ = 'santhosh'

from util import ml_util
from util import NovelMetaGenerator
from nltk.tree import ParentedTree


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


def extractFeaturesFromCoreNLPFiles(core_nlp_files):
    diff_pos = set()
    feature_dict = dict()
    for genre_file_path, genre_file_name in core_nlp_files:
        dictionary = dict()
        with open(genre_file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            line = lines[0]
            line = 'dictionary=' + line
            exec(line)
            curr_file_feature = dict()
            # print genre_file_path, dictionary
            sentences = dictionary[NovelMetaGenerator.SENTENCES]
            for sent in sentences:
                parsetree = sent[NovelMetaGenerator.PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                wordAndtags = t.pos()
                for word, pos in wordAndtags:
                    if pos not in curr_file_feature:
                        curr_file_feature[pos] = 0.0
                    curr_file_feature[pos] += 1.0
                    diff_pos.add(pos)
            feature_dict[genre_file_name] = curr_file_feature

    return feature_dict, diff_pos

def normalize_dist(feature_dict, diff_pos):
    for f in feature_dict:
        feature_dict_for_file = feature_dict[f]
        sum_of_production_rules = sum(feature_dict_for_file.values())
        feature_dict[f] = {k:(feature_dict_for_file[k]/sum_of_production_rules) if k in feature_dict_for_file else 0.0\
                                        for k in diff_pos}
    return feature_dict





def doClassification(allSentencePOS = False):
    meta_dict = readMetaInfo()
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    for genre in meta_dict.keys():
        meta_dict_for_genre = meta_dict[genre]
        file_names = [file_name for file_name in meta_dict_for_genre]
        feature_dict = {file_name: dict() for file_name in file_names}
        if allSentencePOS:
            diff_pos = list(set([pos_tag for file_name in file_names for pos_tag in meta_dict_for_genre[file_name][TAGS]]))
            for file_name in file_names:
                for pos_tag in diff_pos:
                    if pos_tag not in meta_dict_for_genre[file_name][TAGS]:
                        meta_dict_for_genre[file_name][TAGS][pos_tag] = 0.0
                    feature_dict[file_name][pos_tag] = meta_dict_for_genre[file_name][TAGS][pos_tag]
        else:
            core_nlp_files = core_nlp_files_dict[genre]
            feature_dict, diff_pos = extractFeaturesFromCoreNLPFiles(core_nlp_files)
            feature_dict = normalize_dist(feature_dict, diff_pos)

        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict)

        accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print genre, ':', accuracy
