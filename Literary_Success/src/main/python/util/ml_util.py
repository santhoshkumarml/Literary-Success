__author__ = 'santhosh'

import NovelMetaGenerator
import numpy
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.7, rand_idx = True):
    class_wise_genre_file = {NovelMetaGenerator.SUCCESS_PATTERN:[],NovelMetaGenerator.FAILURE_PATTERN:[]}

    for file_name in meta_dict_for_genre:
        if meta_dict_for_genre[file_name][NovelMetaGenerator.CLASS] == NovelMetaGenerator.SUCCESS_PATTERN:
            class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN].append(file_name)
        else:
            class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN].append(file_name)
    total_success_files = len(class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN])
    total_failure_files = len(class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN])
    success_train_size, failure_train_size = int(total_success_files*split), int(total_failure_files*split)

    train_success_idx = set()
    train_failure_idx = set()

    if rand_idx:
        train_success_idx = set(random.sample(xrange(total_success_files), success_train_size))
        train_failure_idx = set(random.sample(xrange(total_failure_files), failure_train_size))
    else:
        for idx in range(0, total_success_files):
            if len(train_success_idx) == success_train_size:
                break
            train_success_idx.add(idx)

        for idx in range(0, total_failure_files):
            if len(train_failure_idx) == failure_train_size:
                break
            train_failure_idx.add(idx)

    train_data, train_result = [], []
    test_data, test_result = [], []

    test_files = []

    for i in range(total_success_files):
            file_name = class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN][i]
            if i in train_success_idx:
                train_data.append([val for val in feature_dict[file_name].values()])
                train_result.append(1)
            else:
                test_data.append([val for val in feature_dict[file_name].values()])
                test_result.append(1)
                test_files.append(file_name)


    for i in range(total_failure_files):
        file_name = class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN][i]
        if i in train_failure_idx:
            train_data.append([val for val in feature_dict[file_name].values()])
            train_result.append(0)
        else:
            test_data.append([val for val in feature_dict[file_name].values()])
            test_result.append(0)
            test_files.append(file_name)

    train_data = numpy.array(train_data)
    train_result = numpy.array(train_result)
    test_data = numpy.array(test_data)
    test_result = numpy.array(test_result)
    return train_data, train_result, test_data, test_result, train_success_idx, train_failure_idx, class_wise_genre_file

def doClassfication(train_data, train_result, test_data, test_result):
    log_r = LinearSVC(C=5)
    log_r.fit(train_data, train_result)
    # accuracy = 0.0
    # for i in range(len(test_data)):
    #     label = int(log_r.predict(test_data[i]))
    #     if label == test_result[i]:
    #         accuracy += 1.0
    # accuracy = accuracy/len(test_data)
    # return accuracy
    mylabel = []
    for k in range(len(test_data)):
        pred = log_r.predict(test_data[k])
        # print pred, test_result[k], test_data[k]
        mylabel.append(pred)

    scores = metrics.precision_recall_fscore_support(test_result, mylabel, labels=[0, 1],\
                                                     pos_label=1, average='binary')
    accuracy = metrics.accuracy_score(test_result, mylabel)

    return scores, accuracy