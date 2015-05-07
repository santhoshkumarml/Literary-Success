import StringIO
__author__ = 'santhosh'

import nltk
from nltk import *
import re
from nltk.tag.stanford import POSTagger

import jsonrpclib
from simplejson import loads
import time
import subprocess
from datetime import datetime
from pywsd import lesk as lsk


nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')

NOVEL_BASE = '/media/santhosh/Data/workspace/nlp_project/novels'
NOVEL_META = 'novel_meta.txt'
CORE_NLP_BASE = '/media/santhosh/Data/workspace/nlp_project/core_nlp'
dataset_pattern = r'[*]+DATASET:.*[*]+'
folder_pattern = r'[*]+.*[*]+'
entry_pattern = r'(SUCCESS|FAILURE).+:.+'
SUCCESS_PATTERN = 'SUCCESS'
FAILURE_PATTERN = 'FAILURE'
POS_PATTERN_FOR_WSD = r'.*[A-Z]+.*'

CORE_NLP_FILE_SUFFIX = '_corenlp1000'
SYNSET_WSD_FILE_SUFFIX = '_wsd1000'
CORE_NLP_TAG_FILES_PATTERN = '.*'+CORE_NLP_FILE_SUFFIX+'.*'
SYNSET_WSD_TAG_PATTERN = '.*'+SYNSET_WSD_FILE_SUFFIX+'.*'


SENTENCES = 'sentences'
PARSE_TREE = 'parsetree'
TXT = 'text'
TUPLES = 'tuples'


KEY_TOKENS = 'FileName|Title|Author|Language|DownloadCount'
LANG_TOKEN = 'Language'
FILE_NAME = 'FileName'
CLASS = 'class'
TAGS = 'TAGS'

def fixMetaInfoRecord(tokens):
    wrong_split_idx = set()
    for i in range(len(tokens)):
        if ':' not in tokens[i]:
            wrong_split_idx.add(i)
    
    mergeable_idx = dict()
    for idx in wrong_split_idx:
        i = 1
        while idx - i in wrong_split_idx:
            i += 1
        
        if idx - i not in mergeable_idx:
            mergeable_idx[idx - i] = []
        mergeable_idx[idx - i].append(idx)
    
    for key in mergeable_idx:
        for idx in mergeable_idx[key]:
            tokens[i] = tokens[i] +','+tokens[idx]
            
    removing_idxs = sorted(list(wrong_split_idx), reverse = True)
    for idx in removing_idxs:
        del tokens[idx]


def processMetaInfoRecord(meta_dict_for_dataset, classification, line):
    line = line.replace(classification+':','')
    tokens = line.split(',')
    
    fixMetaInfoRecord(tokens)
      
    meta_dict_for_file = None
    for token in tokens:
        key,value = token.split(':',1)
        key,value = key.strip(), value.strip()
        if key == FILE_NAME:
            if value in meta_dict_for_dataset:
                print 'Already Present', value
            meta_dict_for_dataset[value] = dict()
            meta_dict_for_file = meta_dict_for_dataset[value]
        else:
            meta_dict_for_file[key] = value
    meta_dict_for_file[CLASS] = classification
    meta_dict_for_file[TAGS] = dict()

def loadInfoFromMetaFile():
    meta_dict = dict()
    with open(os.path.join(NOVEL_BASE,NOVEL_META)) as f:
        lines = f.readlines()
        dataset = None
        for line in lines:
            if re.match(dataset_pattern, line):
                line_strip = line.replace('*','')
                dataset = line_strip.split(':')[1].strip()
                meta_dict[dataset] = dict()
            elif re.match(entry_pattern,line):
                if SUCCESS_PATTERN+':' in line:
                    processMetaInfoRecord(meta_dict[dataset], SUCCESS_PATTERN, line)
                elif FAILURE_PATTERN+':' in line:
                    processMetaInfoRecord(meta_dict[dataset], FAILURE_PATTERN, line)
    return meta_dict

def listGenreWiseFileNames(base_folder, pattern=None):
    genre_folders = [f for f in os.listdir(base_folder) if not os.path.isfile(os.path.join(NOVEL_BASE,f))]
    genre_to_file_list = dict()
    for genre_folder in genre_folders:
        fullPath_to_genre_folder = os.path.join(base_folder, genre_folder)
        multiFolderLevels = [os.path.join(fullPath_to_genre_folder,f)\
                              for f in os.listdir(fullPath_to_genre_folder)\
                              if not os.path.isfile(os.path.join(fullPath_to_genre_folder,f))]
        success_failure_folders = [os.path.join(multiFolderLevel,f)\
                                    for multiFolderLevel in multiFolderLevels\
                                    for f in os.listdir(multiFolderLevel)\
                                    if not os.path.isfile(os.path.join(multiFolderLevel,f))]
        onlyFiles = [(os.path.join(success_failure_folder,f),f)\
                                    for success_failure_folder in success_failure_folders\
                                    for f in os.listdir(success_failure_folder)\
                                    if (os.path.isfile(os.path.join(success_failure_folder,f)) and pattern != None \
                                        and re.match(pattern, f))
                                    or (os.path.isfile(os.path.join(success_failure_folder,f)) and pattern == None)]
        genre_file_dict_key = genre_folder.replace("_",' ')
        genre_to_file_list[genre_file_dict_key] = onlyFiles
        
    return genre_to_file_list

def readGenreBasedFilesAndTagWords(genre_to_file_list, meta_dict, tagger=None):
    for genre in genre_to_file_list:
        meta_dict_for_genre = meta_dict[genre]
        print '--------------------------------------------------------------'
        print 'Number of Files in genre ',genre,' : ',len(meta_dict_for_genre)
        for genre_file_path,genre_file_name in genre_to_file_list[genre]:
            if genre_file_name not in meta_dict_for_genre:
                continue
            pos_tag_dict = dict()
            with open(genre_file_path) as f:
                filelines = f.readlines()
                tokens = [nltk.word_tokenize(line.decode('utf8')) for line in filelines]
                pos_tagged_lines = []
                if tagger != None:
                    pos_tagged_lines = tagger.tag_sents(tokens)
                else:
                    pos_tagged_lines = nltk.pos_tag_sents(tokens)
                for pos_tags in pos_tagged_lines:
                    for word, tag in pos_tags:
                        if tag not in pos_tag_dict:
                            pos_tag_dict[tag] = 0.0
                        pos_tag_dict[tag]+= 1.0
            total_tags = sum(pos_tag_dict.values())
            pos_tag_dict = {key:(pos_tag_dict[key]/total_tags) for key in pos_tag_dict}
            meta_dict_for_genre_file = meta_dict_for_genre[genre_file_name]
            meta_dict_for_genre_file[TAGS] = pos_tag_dict
        print 'Genre ', genre, ' Done'
        print '--------------------------------------------------------------'

def createCORENLPServer():
    corenlp_process = subprocess.Popen(\
        ['python', 'corenlp/corenlp.py','-S','stanford-corenlp-full-2014-08-27/'],\
        stdout=subprocess.PIPE, cwd='/home/santhosh/Downloads/corenlp-python')
    time.sleep(15)
    print 'CORE NLP SERVER STARTED AGAIN AT:', datetime.now()
    return corenlp_process

def readGenreBasedFilesAndRunCoreNLP(genre_to_file_list, meta_dict, genres_to_be_tackled=set()):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for genre in genre_to_file_list:
        if len(genres_to_be_tackled) == 0 or genre not in genres_to_be_tackled:
            continue
        # corenlp_process = createCORENLPServer()
        meta_dict_for_genre = meta_dict[genre]
        print '--------------------------------------------------------------'
        print 'Number of Files in genre ', genre, ' : ', len(meta_dict_for_genre)
        processed = 0
        alreadyProcessed = 0
        for genre_file_path, genre_file_name in genre_to_file_list[genre]:
            core_nlp_path = os.path.dirname(genre_file_path).replace(NOVEL_BASE, CORE_NLP_BASE)
            if not os.path.exists(core_nlp_path):
                os.makedirs(core_nlp_path)
            corenlp_result_file = os.path.join(core_nlp_path,\
                                               genre_file_name.replace('.txt','_corenlp1000.txt'))
            print 'Currently Processing File in genre ', genre, 'File Name:',  corenlp_result_file
            if os.path.isfile(corenlp_result_file):
                alreadyProcessed += 1
                continue
            # if (processed % 2) == 0:
            #     corenlp_process.kill()
            #     corenlp_process = createCORENLPServer()
            print 'Already Processed File Count:', processed+alreadyProcessed
            with open(genre_file_path) as f:
                filelines = f.read()
                filelines = filelines.replace('\r\n', ' ')
                filelines = filelines.replace('\n', ' ')
                sents = sent_detector.tokenize(filelines.decode('utf-8').strip())
                # print type(filelines), len(filelines), filelines, genre_file_path
                # filelines = filelines[0]
                # print filelines
                sents = sents[:1000]
                string = ''
                for line in sents:
                    server = jsonrpclib.Server("http://localhost:8080")
                    result = loads(server.parse(line))
                    string += str(result)+'\n'
                with open(corenlp_result_file, 'w') as f1:
                    f1.write(string)
            processed += 1
        # corenlp_process.kill()

def extractMetaDataAndPOSTagsDistributions():
    start_time = datetime.now()
    meta_dict = loadInfoFromMetaFile()
    genre_to_file_list = listGenreWiseFileNames()
    train_data = nltk.corpus.treebank.tagged_sents()
    unigramTagger = UnigramTagger(train_data, backoff=nltk.DefaultTagger('NN'))
    bigramTagger = BigramTagger(train_data, backoff=unigramTagger)
    stanford_tagger = POSTagger('/media/santhosh/Data/workspace/nlp_project/models/english-left3words-distsim.tagger',
                    '/media/santhosh/Data/workspace/nlp_project/stanford-postagger.jar')
    TAGGERS = {'UNIGRAM': unigramTagger, 'BIGRAM':bigramTagger, 'STANFORD':stanford_tagger}
    readGenreBasedFilesAndTagWords(genre_to_file_list, meta_dict, None)
    with open('../../../../novel_meta_pos_nltk_tagger.meta', 'w') as f:
        f.write(str(meta_dict))
    end_time = datetime.now()
    print 'Total Time', end_time - start_time

def extractMetaDataAndTagCoreNLP(genres=None):
    if genres == None:
        genres = set(['Adventure Stories', 'Love Stories', 'Poetry', 'Mystery',\
                      'Short Stories', 'Fiction', 'Science Fiction',\
                      'Historical Fiction'])
    start_time = datetime.now()
    meta_dict = loadInfoFromMetaFile()
    genre_to_file_list = listGenreWiseFileNames(NOVEL_BASE)
    readGenreBasedFilesAndRunCoreNLP(genre_to_file_list, meta_dict, genres)
    end_time = datetime.now()
    print 'Total Time', end_time - start_time


def readGenreFilesAndTagWordsForSenses(core_nlp_files):
    for genre_file_path, genre_file_name in core_nlp_files:
        dictionary = dict()
        print genre_file_path
        with open(genre_file_path) as f:
            synset_wsd_file = genre_file_path.replace(CORE_NLP_FILE_SUFFIX, SYNSET_WSD_FILE_SUFFIX)
            if os.path.exists(synset_wsd_file):
                continue
            lines = f.readlines()[:100]
            output = []
            for line in lines:
                line = 'dictionary=' + line
                exec(line)
                sentences = dictionary[SENTENCES]
                sent = sentences[0]
                parsetree = sent[PARSE_TREE]
                t = ParentedTree.fromstring(parsetree)
                sentence_result = []
                txt = sent[TXT]
                for word, pos in t.pos():
                    if re.match(POS_PATTERN_FOR_WSD, pos) and pos not in ['DT', 'CC', 'CD']:
                        ranked_synsets = lsk.adapted_lesk(unicode(txt), unicode(word))
                        result = (word, ranked_synsets)
                        sentence_result.append(result)
                output.append(sentence_result)

            with open(synset_wsd_file, 'w') as f1:
                f1.write(str(output))


def extractSysetDistributionForWORDS():
    start_time = datetime.now()
    meta_dict = loadInfoFromMetaFile()
    core_nlp_files_dict = listGenreWiseFileNames(CORE_NLP_BASE, CORE_NLP_TAG_FILES_PATTERN)
    for genre in ['Adventure Stories', 'Love Stories']:
        readGenreFilesAndTagWordsForSenses(core_nlp_files_dict[genre])
    end_time = datetime.now()
    print 'Total Time', end_time - start_time

# extractMetaDataAndTagCoreNLP()