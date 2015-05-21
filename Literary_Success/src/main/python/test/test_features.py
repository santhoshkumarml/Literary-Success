__author__ = 'santhosh'

from util import NovelMetaGenerator
from util import ml_util
from feature_extractor import POSFeatureUtil
from feature_extractor import SyntaticTreeFeaturesUtil
from feature_extractor import DeepSyntacticFeatureUtil
from feature_extractor import WordSenseAmbiguityFeatureUtil
import numpy


def plotDataPoints(feature_dict, genre, train_success_idx, train_failure_idx, class_wise_genre_file):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title('Average Entropy Plot for '+genre)
    plt.ylabel('Average Entropy')
    plt.xlabel('Novel')
    ax = fig.add_subplot(1, 1, 1)
    cnt = 1
    for k in class_wise_genre_file:
        for idx in range(len(class_wise_genre_file[k])):
            f = class_wise_genre_file[k][idx]
            entropy = feature_dict[f]['AVG_ENTROPY']
            if k == NovelMetaGenerator.SUCCESS_PATTERN:
                if idx not in train_success_idx:
                    ax.plot(cnt+1, entropy, 'go')
                    cnt += 1
            else:
                if idx not in train_failure_idx:
                    ax.plot(cnt+1, entropy, 'ro')
                    cnt += 1
    plt.show()

def plotSenseDistribution(feature_dict, genre, class_wise_genre_file):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title('Sense Distribution for '+genre)
    plt.ylabel('Number Of Senses')
    plt.xlabel('Novel')
    ax = fig.add_subplot(1, 1, 1)
    width = 0.20

    success_feature_vals = []
    failure_feature_vals = []

    colors = ['r', 'b', 'g', 'y', 'c', 'm']

    for success_file in class_wise_genre_file[NovelMetaGenerator.SUCCESS_PATTERN]:
        feature_vals = feature_dict[success_file].values()
        success_feature_vals.append(feature_vals)

    for failure_file in class_wise_genre_file[NovelMetaGenerator.FAILURE_PATTERN]:
        feature_vals = feature_dict[failure_file]
        failure_feature_vals.append(feature_vals)

    rects = []
    s_cnt = len(success_feature_vals)
    f_cnt = len(failure_feature_vals)

    s_ind = numpy.arange(0, s_cnt, 1)
    f_ind = numpy.arange(s_cnt, (s_cnt + f_cnt), 1)

    for i in range(len(success_feature_vals)):
        rect = ax.bar(s_ind + (width * i), success_feature_vals[i], width=width, color=colors[i])
        rects.append(rect)

    for j in range(len(failure_feature_vals)):
        rect = ax.bar(f_ind + (width * j), failure_feature_vals[j], width=width, color=colors[j])
        rects.append(rect)

    ax.set_xticks(numpy.concatenate([s_ind, f_ind]) + width)
    ax.set_xticklabels(['S' if idx < s_cnt else 'F' for idx in range(s_cnt+f_cnt)])
    # ax.legend(rect)

    plt.show()



def testSenseDistribution(genres = None):
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.SYNSET_WSD_TAG_PATTERN)
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    if not genres:
        genres = NovelMetaGenerator.ALL_GENRES

    for genre in genres:
        core_nlp_files = core_nlp_files_dict[genre]
        meta_dict_for_genre = meta_dict[genre]
        feature_dict = WordSenseAmbiguityFeatureUtil.extractSenseDistributionFeatures(core_nlp_files)
        train_data, train_result, test_data, test_result, train_success_idx, train_failure_idx, class_wise_genre_file =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.7, rand_idx=False)
        # plotSenseDistribution(feature_dict, genre, class_wise_genre_file)
        scores = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print scores

def testPOSFeatures(genres=None):
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    if not genres:
        genres = NovelMetaGenerator.ALL_GENRES

    for genre in genres:
        core_nlp_files = core_nlp_files_dict[genre]
        meta_dict_for_genre = meta_dict[genre]
        feature_dict = POSFeatureUtil.extractPOSFeaturesFromCoreNLPFiles(core_nlp_files)
        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.8)
        scores = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print scores

def testProductionFeatures(genres=None):
    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    if not genres:
        genres = NovelMetaGenerator.ALL_GENRES

    for genre in genres:
        core_nlp_files = core_nlp_files_dict[genre]
        meta_dict_for_genre = meta_dict[genre]
        feature_dict = SyntaticTreeFeaturesUtil.extractSyntacticFeatures(core_nlp_files)
        train_data, train_result, test_data, test_result =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.8)
        scores = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print scores


def testDeepSyntacticFeatures(genres=None, features=None):

    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.CORE_NLP_TAG_FILES_PATTERN)
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    if not genres:
        genres = NovelMetaGenerator.ALL_GENRES

    if not features:
        features = DeepSyntacticFeatureUtil.ALL_DEEP_SYNTACTIC_FEATURES


    for genre in genres:
        core_nlp_files = core_nlp_files_dict[genre]
        meta_dict_for_genre = meta_dict[genre]
        feature_dict = DeepSyntacticFeatureUtil.extractDeepSyntaticFeature(core_nlp_files, features)
        for f in feature_dict:
            print meta_dict_for_genre[f][NovelMetaGenerator.CLASS]
            print feature_dict[f]
            print '--------------------------'
        train_data, train_result, test_data, test_result, train_success_idx, train_failure_idx, class_wise_genre_file =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.7, rand_idx=True)
        scores = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print scores



def testAmbiguity(genres=None):

    core_nlp_files_dict = NovelMetaGenerator.listGenreWiseFileNames(NovelMetaGenerator.CORE_NLP_BASE,\
                                                                    NovelMetaGenerator.SYNSET_WSD_TAG_PATTERN)
    meta_dict = NovelMetaGenerator.loadInfoFromMetaFile()
    if not genres:
        genres = NovelMetaGenerator.ALL_GENRES


    for genre in genres:
        core_nlp_files = core_nlp_files_dict[genre]
        meta_dict_for_genre = meta_dict[genre]
        feature_dict = WordSenseAmbiguityFeatureUtil.extractSenseEntropyFeature(core_nlp_files)
        train_data, train_result, test_data, test_result, train_success_idx, train_failure_idx, class_wise_genre_file =\
            ml_util.splitTrainAndTestData(meta_dict_for_genre, feature_dict, split=0.7, rand_idx=False)
        plotDataPoints(feature_dict, genre, train_success_idx, train_failure_idx, class_wise_genre_file)
        scores = ml_util.doClassfication(train_data, train_result, test_data, test_result)
        print scores




genres_to_test = set(['Adventure Stories', 'Love Stories'])
genres_to_test = set(['Adventure Stories'])

# print 'POS Features'
# testPOSFeatures(genres=genres_to_test)
#
# print 'Production Features'
# testProductionFeatures(genres=genres_to_test)
#
# print 'Deep Syntactic Tree Structure Features'
# testDeepSyntacticFeatures(genres=genres_to_test,\
#                           features=set([DeepSyntacticFeatureUtil.HEIGHT,
#                                         DeepSyntacticFeatureUtil.HORIZONTAL_TREE_IMBALANCE,\
#                                         DeepSyntacticFeatureUtil.VERTICAL_TREE_IMBALANCE]))

# testDeepSyntacticFeatures(genres=genres_to_test,\
#                           features=set([DeepSyntacticFeatureUtil.COMPLEX_COMPOUND_FEATURE,\
#                                         DeepSyntacticFeatureUtil.LOOSE_PERIODIC_FEATURE]))

# print 'Word Sense Ambiguity Features'
# testAmbiguity(genres=genres_to_test)


def plotDeepSyntactic():
    with open('/home/santhosh/sample1') as f:
        lines = f.readlines()
        curr_type = None
        data_dict = dict()
        import matplotlib.pyplot as plt
        import numpy

        fnct = 0
        s_cnt, f_cnt = 0, 0
        thres = 10
        width = 0.20
        jump = 1
        fig = plt.figure(figsize=(20, 20))
        plt.title('Sentence Type 2: Loose/Periodic')
        plt.xlabel('Novel Labels')
        plt.ylabel('Distribution of sentence types')
        ax = fig.add_subplot(1, 1, 1)

        features = [DeepSyntacticFeatureUtil.COMPLEX_COMPOUND, \
                                             DeepSyntacticFeatureUtil.COMPLEX, \
                                             DeepSyntacticFeatureUtil.COMPOUND, \
                                             DeepSyntacticFeatureUtil.SIMPLE]
        features = [DeepSyntacticFeatureUtil.LOOSE, DeepSyntacticFeatureUtil.PERIODIC]

        success_feature_vals = [[] for k in features]

        failure_feature_vals = [[] for k in features]

        feature_names = []
        colors = ['r', 'b', 'g', 'y', 'c', 'm']

        for line in lines:
            if s_cnt >= thres and f_cnt >= thres:
                break

            line = line.strip()

            if line == NovelMetaGenerator.SUCCESS_PATTERN:
                curr_type = NovelMetaGenerator.SUCCESS_PATTERN
                s_cnt += 1
            elif line == NovelMetaGenerator.FAILURE_PATTERN:
                curr_type = NovelMetaGenerator.FAILURE_PATTERN
                f_cnt += 1
            elif '----' not in line:
                if (curr_type == NovelMetaGenerator.SUCCESS_PATTERN and s_cnt > thres+1) or \
                        (curr_type == NovelMetaGenerator.FAILURE_PATTERN and f_cnt > thres+1):
                    continue

                exec ('data_dict=' + line)
                vals = [(k, v) for k, v in data_dict.items() if k in features]

                feature_names = [k for k, v in vals if k in features]

                for i in range(len(features)):
                    if curr_type == NovelMetaGenerator.SUCCESS_PATTERN and len(success_feature_vals[i]) < thres:
                        success_feature_vals[i].append(vals[i][1])
                    else:
                        if len(failure_feature_vals[i]) < thres:
                            failure_feature_vals[i].append(vals[i][1])

                fnct += jump

        print fnct, s_cnt, f_cnt
        s_cnt, f_cnt = (thres, thres)

        s_ind = numpy.arange(0, s_cnt * jump, jump)
        f_ind = numpy.arange(s_cnt * jump, (s_cnt + f_cnt) * jump, jump)

        rects = []

        for i in range(len(feature_names)):
            print len(success_feature_vals[i]), len(failure_feature_vals[i])

        for i in range(len(success_feature_vals)):
            rect = ax.bar(s_ind + (width * i), success_feature_vals[i], width=width, color=colors[i])
            rects.append(rect)

        for j in range(len(failure_feature_vals)):
            rect = ax.bar(f_ind + (width * j), failure_feature_vals[j], width=width, color=colors[j])
            rects.append(rect)

        ax.set_xticks(numpy.concatenate([s_ind, f_ind]) + width)
        ax.set_xticklabels(['S' if idx < s_cnt else 'F' for idx in range(fnct)])
        ax.legend(rects, feature_names)

        plt.show()

# testAmbiguity(genres=set(['Adventure Stories']))
testSenseDistribution(genres=set(['Adventure Stories']))