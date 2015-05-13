
__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import ml_util
import math
from nltk.corpus.reader import Synset
import re

def calculate_Entropy(dist):
    entropy_list = [-p*math.log(p, 2) for p in dist if p != 0]
    if len(entropy_list) > 0:
        return sum(entropy_list)
    else:
        return 0

def extractConnotationFeatures(conn_files):
    feature_dict = dict()
    for genre_file_path, genre_file_name in conn_files:
        all_entropy = []
        f = open(genre_file_path)
        lines = f.readlines()
        f.close()
        assert len(lines) == 1
        line = lines[0]
        data = []
        # line = re.sub(r'Synset\(.*?\)','\'-WSD-\'', line)
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
        key = genre_file_name.replace(NovelMetaGenerator.SYNSET_WSD_FILE_SUFFIX, '')
        feature_dict[key] = {'AVG_ENTROPY': avg_entropy}
    return feature_dict
