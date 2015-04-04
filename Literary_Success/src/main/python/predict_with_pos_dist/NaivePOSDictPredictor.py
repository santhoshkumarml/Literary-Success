__author__ = 'santhosh'

content = 'meta_dict='
import numpy
import random
from sklearn.cluster import k_means
from sklearn.linear_model.logistic import LogisticRegression

CLASS = 'class'
TAGS = 'TAGS'
SUCCESS_PATTERN = 'SUCCESS'
FAILURE_PATTERN = 'FAILURE'

def readMetaInfo():
    content = 'meta_dict='
    with open('../../../../novel_meta_pos.meta', 'r') as f:
        meta_dict = dict()
        content = content + f.readline()
        exec (content)
        return meta_dict

def prepareData(meta_dict, split = 0.9):
    genre_wise_train_data = dict()
    genre_wise_test_data = dict()
    diff_genres = meta_dict.keys()
    for genre in diff_genres:
        meta_dict_for_genre = meta_dict[genre]
        file_names = [file_name for file_name in meta_dict_for_genre]
        diff_pos = list(set([pos_tag for file_name in file_names for pos_tag in meta_dict_for_genre[file_name][TAGS]]))

        n_samples = len(file_names)
        n_features = len(diff_pos)
        data = numpy.zeros(shape=(n_samples, n_features))

        sample_idx = 0
        for file_name in file_names:
            feature_idx = 0
            for pos_tag in diff_pos:
                if pos_tag not in meta_dict_for_genre[file_name][TAGS]:
                    meta_dict_for_genre[file_name][TAGS][pos_tag] = 0.0
                data[sample_idx][feature_idx] = meta_dict_for_genre[file_name][TAGS][pos_tag]
                feature_idx += 1
            sample_idx += 1

        class_wise_genre_file = {SUCCESS_PATTERN:[],FAILURE_PATTERN:[]}
        for file_name in meta_dict_for_genre:
            if meta_dict_for_genre[file_name][CLASS] == SUCCESS_PATTERN:
                class_wise_genre_file[SUCCESS_PATTERN].append(file_name)
            else:
                class_wise_genre_file[FAILURE_PATTERN].append(file_name)
        total_success_files = len(class_wise_genre_file[SUCCESS_PATTERN])
        total_failure_files = len(class_wise_genre_file[FAILURE_PATTERN])
        success_train_size,failure_train_size = int(total_success_files*split), int(total_failure_files*split)
        success_test_size,failure_test_size = (total_success_files-success_train_size),(total_failure_files-failure_train_size)

        random_train_success_idx = set(random.sample(xrange(total_success_files), success_train_size))
        #random_test_success_idx = set(range(total_success_files)) - random_train_success_idx

        random_train_failure_idx = set(random.sample(xrange(total_failure_files), failure_train_size))
        #random_test_failure_idx = set(range(total_failure_files)) - random_train_failure_idx

        genre_wise_train_data[genre] = ([],[])
        genre_wise_test_data[genre] = ([],[])

        for i in range(total_success_files):
            if i in random_train_success_idx:
                genre_wise_train_data[genre][0].append(class_wise_genre_file[SUCCESS_PATTERN][i])
                genre_wise_train_data[genre][1].append(1)
            else:
                genre_wise_test_data[genre][0].append(class_wise_genre_file[SUCCESS_PATTERN][i])
                genre_wise_test_data[genre][1].append(1)
        for i in range(total_failure_files):
            if i in random_train_failure_idx:
                genre_wise_train_data[genre][0].append(class_wise_genre_file[FAILURE_PATTERN][i])
                genre_wise_train_data[genre][1].append(0)
            else:
                genre_wise_test_data[genre][0].append(class_wise_genre_file[FAILURE_PATTERN][i])
                genre_wise_test_data[genre][1].append(0)

    return genre_wise_train_data, genre_wise_test_data


def makeClassificationAndMeasureAccuracy(genre_wise_train_data, genre_wise_test_data, meta_dict):
    accuracy_for_genre = dict()
    for genre in genre_wise_train_data:
        meta_dict_for_genre = meta_dict[genre]
        train_data, train_result = genre_wise_train_data[genre]
        test_data, test_result = genre_wise_test_data[genre]
        train_data = [list(meta_dict_for_genre[file_name][TAGS].values()) for file_name in train_data]
        test_data = [list(meta_dict_for_genre[file_name][TAGS].values()) for file_name in test_data]
        log_r = LogisticRegression()
        log_r.fit(train_data, train_result)
        accuracy = 0.0
        for i in range(len(test_data)):
            label = int(log_r.predict(test_data[i]))
            if label == test_result[i]:
                accuracy += 1.0
        accuracy = accuracy/len(test_data)
        accuracy_for_genre[genre] = accuracy
    return  accuracy_for_genre







        # def tryclustering():
        # kmeans = KMeans(n_clusters=2)
        # predicted_values = kmeans.fit_predict(data)
        #
        # ONE = {SUCCESS_PATTERN:[], FAILURE_PATTERN:[]}
        # ZERO = {SUCCESS_PATTERN:[], FAILURE_PATTERN:[]}
        # for sample_idx in range(n_samples):
        # file_name = file_names[sample_idx]
        #     if predicted_values[sample_idx]:
        #         if meta_dict_for_genre[file_name][CLASS] == SUCCESS_PATTERN:
        #             ONE[SUCCESS_PATTERN].append(file_name)
        #         else:
        #             ONE[FAILURE_PATTERN].append(file_name)
        #     else:
        #         if meta_dict_for_genre[file_name][CLASS] == SUCCESS_PATTERN:
        #             ZERO[SUCCESS_PATTERN].append(file_name)
        #         else:
        #             ZERO[FAILURE_PATTERN].append(file_name)
        # print '-------------------------------------------------------------'
        # print 'GENRE:', genre
        # print len(ONE[SUCCESS_PATTERN]), len(ONE[FAILURE_PATTERN])
        # print len(ZERO[SUCCESS_PATTERN]), len(ZERO[FAILURE_PATTERN])
        # print '-------------------------------------------------------------'

meta_dict = readMetaInfo()
genre_wise_train_data, genre_wise_test_data = prepareData(meta_dict, split=0.8)
genre_wise_accuracy = makeClassificationAndMeasureAccuracy(genre_wise_train_data, genre_wise_test_data, meta_dict)
print genre_wise_accuracy