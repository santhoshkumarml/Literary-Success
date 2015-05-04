__author__ = 'santhosh'

import NovelMetaGenerator
import numpy
import random
from sklearn.linear_model import LogisticRegression

def splitTrainAndTestData(meta_dict_for_genre, feature_dict, split = 0.7):
    file_names = [file_name for file_name in meta_dict_for_genre]

    n_samples = len(file_names)
    n_features = len(feature_dict[file_name].values())
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

    train_data, train_result = [],[]
    test_data, test_result = [],[]

    for i in range(total_success_files):
            file_name = class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN][i]
            if i in random_train_success_idx:
                train_data.append(list(feature_dict[file_name].values()))
                train_result.append(1)
            else:
                test_data.append(list(feature_dict[file_name].values()))
                test_result.append(1)

    for i in range(total_failure_files):
        file_name = class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN][i]
        if i in random_train_failure_idx:
            train_data.append(list(feature_dict[file_name].values()))
            train_result.append(0)
        else:
            test_data.append(list(feature_dict[file_name].values()))
            test_result.append(0)

    return train_data, train_result, test_data, test_result



def doClassfication(train_data, train_result, test_data, test_result):
    log_r = LogisticRegression()
    log_r.fit(train_data, train_result)
    accuracy = 0.0
    for i in range(len(test_data)):
        label = int(log_r.predict(test_data[i]))
        if label == test_result[i]:
            accuracy += 1.0
    accuracy = accuracy/len(test_data)
    return accuracy