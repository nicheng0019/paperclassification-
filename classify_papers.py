# -*- coding:utf8 -*-
import os
import re
import codecs
import shutil
import numpy as np
import fnmatch
from textblob import TextBlob
#from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer

from utils import *

dict_vec = {}

def get_normalize_name(originalname):
    lem = WordNetLemmatizer()

    pattern = '_[0-9]+.[0-9]+(.*)pdf'
    searchobj = re.search(pattern, originalname)
    result = ".pdf"
    if searchobj != None:
        result = searchobj.group()

    name = originalname.replace(result, "")

    name = clean_string(name)
    words = re.split(r"[;._\s]", name)

    try:
        words_str = ""
        for word in words:
            lower_word = word.lower()
            correct_word = TextBlob(lower_word).correct().words[0]
            lemmatize_word = lem.lemmatize(correct_word, "n")
            words_str = words_str + lemmatize_word + " "
    except:
        print(originalname)
        return None

    return words_str

def get_feature_by_filename(filename, word_dict):
    feature = np.zeros((1, len(word_dict.keys())))
    words = filename.strip().split(" ")
    for word in words:
        if word in word_dict.keys():
            feature[0, word_dict[word]] = feature[0, word_dict[word]] + 1

    return feature

def get_words_feature(words):
    global dict_vec
    feature = []
    for word in words:
        if word not in dict_vec.keys():
            continue

        feature.append(dict_vec[word])

    return np.array(feature)

def get_paper_content(filename):
    with codecs.open(filename, mode="r", encoding="utf-8") as ftxt:
        content = ftxt.readlines()

    if len(content) == 0:
        return ""

    content = content[0]
    words = content.split(" ")
    wordsnew = []
    for word in words:
        word = word.lower()
        if check_invalid_word(word):
            continue

        wordsnew.append(word)

    print(wordsnew)

    return wordsnew

def get_paper_feature(fnpath):
    content = get_paper_content(fnpath)

    get_words_feature(content)



def classify_papers(rootdir, num_clusters=5, featuretype="content"):
    for fn in os.listdir(rootdir):
        fnpath = os.path.join(rootdir, fn)
        if os.path.isdir(fnpath):
            continue

        if featuretype == "name":
            names = []
            name = get_normalize_name(fn)
            if name != None:
                names.append(unicode(name))

            count_v1 = CountVectorizer(max_df=0.8, min_df=0.01)
            counts_train = count_v1.fit_transform(names)

            word_dict = {}
            for index, word in enumerate(count_v1.get_feature_names()):
                word_dict[word] = index

            features = np.zeros((len(names), len(word_dict.keys())))
            for index, name in enumerate(names):
                features[index] = get_feature_by_filename(name, word_dict)[0]

        else:
            features = []
            feature = get_paper_feature(fnpath)

            if feature != None:
                features.append(feature)

            features = np.array(features)

    km = KMeans(n_clusters=num_clusters).fit(features)
    paper_labels = km.labels_

    print(paper_labels)

    return
    for fn in os.listdir(rootdir):
        fnpath = os.path.join(rootdir, fn)
        if os.path.isdir(fnpath):
            continue

        normal_name = get_normalize_name(fn)

        if normal_name != None:
            feature = get_feature_by_filename(normal_name, word_dict)
            predictresult = km.predict(feature)
            dstdir = os.path.join(rootdir, str(predictresult[0]))
        else:
            dstdir = os.path.join(rootdir, "notsure")

        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        shutil.copy(os.path.join(rootdir, fn), os.path.join(dstdir, fn))

def init_dict_vec(dictfile):
    global dict_vec
    with open(dictfile, "r") as fdict:
        lines = fdict.readlines()

    for line in lines:
        datas = line.split(" ")
        word = datas[0]
        vec = datas[1:]
        vec = map(eval, vec)

        dict_vec[word] = np.array(vec)

    print(dict_vec["and"])


if __name__=='__main__':
    dictfile = r"resource/glove.6B.50d.txt"
    init_dict_vec(dictfile)

    filedir = r"content"
    num_clusters = 10
    classify_papers(filedir, num_clusters=num_clusters)