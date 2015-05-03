__author__ = 'santhosh'

from util import NovelMetaGenerator
from nltk.tree import ParentedTree
import numpy
import random
from sklearn.linear_model import LogisticRegression

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
            production_dict_for_files[genre_file_name.replace('_corenlp1000.txt', '.txt')] = production_dict
    return production_dict_for_files, diff_productions


def normalize_dist(production_dict_for_files, diff_productions):
    for f in production_dict_for_files:
        prod_dict_for_file = production_dict_for_files[f]
        sum_of_production_rules = sum(prod_dict_for_file.values())
        production_dict_for_files[f] = {k:(prod_dict_for_file[k]/sum_of_production_rules) if k in prod_dict_for_file else 0.0\
                                        for k in diff_productions.keys()}
    return production_dict_for_files


def splitTrainAndTestData(meta_dict_for_genre, production_dict_for_files, split = 0.7):
    file_names = [file_name for file_name in meta_dict_for_genre]


    n_samples = len(file_names)
    n_features = len(production_dict_for_files[file_name].values())
    data = numpy.zeros(shape=(n_samples, n_features))


    class_wise_genre_file = {NovelMetaGenerator.SUCCESS_PATTERN:[],NovelMetaGenerator.FAILURE_PATTERN:[]}

    for file_name in meta_dict_for_genre:
        if meta_dict_for_genre[file_name][NovelMetaGenerator.CLASS] == NovelMetaGenerator.SUCCESS_PATTERN:
            class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN].append(file_name)
        else:
            class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN].append(file_name)
    total_success_files = len(class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN])
    total_failure_files = len(class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN])
    success_train_size, failure_train_size = int(total_success_files*split), int(total_failure_files*split)

    random_train_success_idx = set(random.sample(xrange(total_success_files), success_train_size))

    random_train_failure_idx = set(random.sample(xrange(total_failure_files), failure_train_size))

    train_data = ([],[])
    test_data = ([],[])

    for i in range(total_success_files):
            file_name = class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN][i]
            if i in random_train_success_idx:
                train_data[0].append(list(production_dict_for_files[file_name].values()))
                train_data[1].append(1)
            else:
                test_data[0].append(list(production_dict_for_files[file_name].values()))
                test_data[1].append(1)

    for i in range(total_failure_files):
        file_name = class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN][i]
        if i in random_train_failure_idx:
            train_data[0].append(list(production_dict_for_files[file_name].values()))
            train_data[1].append(0)
        else:
            test_data[0].append(list(production_dict_for_files[file_name].values()))
            test_data[1].append(0)

    return train_data, test_data




def doClassification():
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE)
    novel_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE)
    for genre in core_nlp_files_dict:
        if genre == 'Science Fiction' or genre == 'Short Stories':
            continue
        meta_dict_for_genre = meta_dict[genre]
        core_nlp_files = core_nlp_files_dict[genre]
        production_dict_for_files, diff_productions = getConsituentTreeDistribution(core_nlp_files)
        production_dict_for_files = normalize_dist(production_dict_for_files, diff_productions)
        train_data, test_data = splitTrainAndTestData(meta_dict_for_genre, production_dict_for_files)
        log_r = LogisticRegression()
        train_data, train_result = train_data
        test_data, test_result = test_data
        log_r.fit(train_data, train_result)
        accuracy = 0.0
        for i in range(len(test_data)):
            label = int(log_r.predict(test_data[i]))
            if label == test_result[i]:
                accuracy += 1.0
        accuracy = accuracy/len(test_data)
        print genre, ':', accuracy


doClassification()