__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree

def readCoreNLPFileAndReturnTree(core_nlp_file):
    genre_file_name, genre_file_path = core_nlp_file
    dictionary = dict()
    trees = []
    with open(genre_file_path) as f:
        lines = f.readlines()[:100]
        for line in lines:
            line = 'dictionary=' + line
            exec(line)
            sentences = dictionary[NovelMetaGenerator.SENTENCES]
            for sent in sentences:
                parsetree = sent[NovelMetaGenerator.PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                trees.append(t)
    return trees
