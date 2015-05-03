__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.corpus.reader import Synset
import math

def calculate_Entropy(dist):
    entropy_list = [-p*math.log(p) for p in dist if p != 0]
    if len(entropy_list) > 0:
        return sum(entropy_list)
    else:
        return 0

def extractFeatures():
    conn_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                 NovelMetaGenerator.SYNSET_WSD_TAG_PATTERN)
    fs = conn_files_dict['Adventure Stories']
    feature_dict = dict()

    for genre_file_path, genre_file_name in fs:
        all_entropy = []
        f = open(genre_file_path)
        lines = f.readlines()
        f.close()
        assert len(lines) == 1
        line = lines[0]
        data = []
        line = 'data='+line
        exec(line)
        for line in data:
         for word, synset in line:
            if synset == None:
                continue
            entropy = calculate_Entropy([p for p, syn in synset])
            if entropy > 0:
                all_entropy.append(entropy)
        avg_entropy = 0.0
        if len(all_entropy) > 0:
            avg_entropy = sum(all_entropy)/len(all_entropy)
        feature_dict[genre_file_name] = avg_entropy
    return feature_dict

print extractFeatures()


