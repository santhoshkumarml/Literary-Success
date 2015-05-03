__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree

SENTENCES = 'sentences'
PARSE_TREE = 'parsetree'
TXT = 'text'
TUPLES = 'tuples'

def getConsituentTreeDistribution(core_nlp_files):
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
            sentences = dictionary[SENTENCES]
            for sent in sentences:
                parsetree = sent[PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                prod = t.productions()
                if prod not in diff_productions:
                    diff_productions[prod] = 0.0
                if prod not in production_dict:
                    production_dict[prod] = 0.0
                diff_productions[prod] += 1.0
                production_dict[prod] += 1.0
            production_dict_for_files[genre_file_name] = production_dict

    for f in production_dict_for_files:
        prod_dict_for_file = production_dict_for_files[f]
        production_dict_for_files[f] = {prod_dict_for_file[k]/diff_productions[k] if k in prod_dict_for_file else 0.0\
                                        for k in diff_productions.keys()}
    return production_dict_for_files


def doClassification():
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE)

    for genre in core_nlp_files_dict:
        core_nlp_files = core_nlp_files_dict[genre]
        prod_dict_for_files = getConsituentTreeDistribution(core_nlp_files)
        print prod_dict_for_files

doClassification()