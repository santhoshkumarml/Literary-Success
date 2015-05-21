
__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import ml_util
import math
from nltk.corpus.reader import Synset
import re


def normalize_dist(feature_dict, max_sense_count):
    for f in feature_dict:
        sense_counts_for_file = feature_dict[f]
        sum_of_sense_count = sum(sense_counts_for_file.values())
        sense_counts_for_file = {k:(sense_counts_for_file[k]/sum_of_sense_count)\
                                        if k in sense_counts_for_file else 0.0 \
                                        for k in range(1, max_sense_count)}
        feature_dict[f] = sense_counts_for_file
    return feature_dict

def calculate_Entropy(dist):
    entropy_list = [-p*math.log(p, 2) for p in dist if p != 0]
    if len(entropy_list) > 0:
        return sum(entropy_list)
    else:
        return 0



def extractSenseDistributionFeatures(conn_files):
    feature_dict = dict()
    max_sense_count = -1
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
        key = genre_file_name.replace(NovelMetaGenerator.SYNSET_WSD_FILE_SUFFIX, '')
        feature_dict[key] = dict()
        for line in data:
            for word, synset in line:
                if synset == None:
                    continue
                number_of_senses = len(synset)
                if number_of_senses > max_sense_count:
                    max_sense_count = number_of_senses
                if number_of_senses not in feature_dict[key]:
                    feature_dict[key][number_of_senses] = 0.0
                feature_dict[key][number_of_senses] += 1.0
    normalize_dist(feature_dict, max_sense_count)
    return feature_dict

def extractSenseEntropyFeature(conn_files):
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
