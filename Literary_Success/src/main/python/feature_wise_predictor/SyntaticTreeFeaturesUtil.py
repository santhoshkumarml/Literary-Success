__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import ml_util
from nltk.tree import ParentedTree
from util import data_reader


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
    for core_nlp_file in core_nlp_files:
        genre_file_name, genre_file_path = core_nlp_file
        production_dict = dict()
        trees = data_reader.readCoreNLPFileAndReturnTree(core_nlp_file)
        for t in trees:
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
