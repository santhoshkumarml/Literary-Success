__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import ml_util
from nltk.tree import ParentedTree


def normalize_dist(production_dict_for_files, diff_productions):
    for f in production_dict_for_files:
        prod_dict_for_file = production_dict_for_files[f]
        sum_of_production_rules = sum(prod_dict_for_file.values())
        production_dict_for_files[f] = {k:(prod_dict_for_file[k]/sum_of_production_rules)\
                                        if k in prod_dict_for_file else 0.0 \
                                        for k in diff_productions.keys()}
    return production_dict_for_files


def extractSyntacticFeatures(core_nlp_files):
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
            sentences = dictionary[NovelMetaGenerator.SENTENCES]
            for sent in sentences:
                parsetree = sent[NovelMetaGenerator.PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                prods = t.productions()
                for prod in prods:
                    if prod not in diff_productions:
                        diff_productions[prod] = 0.0
                    if prod not in production_dict:
                        production_dict[prod] = 0.0
                    diff_productions[prod] += 1.0
                    production_dict[prod] += 1.0
            key = genre_file_name.replace(NovelMetaGenerator.CORE_NLP_FILE_SUFFIX, '')
            production_dict_for_files[key] = production_dict

    production_dict_for_files = normalize_dist(production_dict_for_files, diff_productions)
    return production_dict_for_files




def doClassification():
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    for genre in core_nlp_files_dict:
        if genre == 'Science Fiction' or genre == 'Short Stories':
            continue
        meta_dict_for_genre = meta_dict[genre]
        core_nlp_files = core_nlp_files_dict[genre]
        feature_dict = extractSyntacticFeatures(core_nlp_files)
        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict)
        accuracy = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print genre, ':', accuracy