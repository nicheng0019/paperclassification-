# -*- coding:utf8 -*-
import os
import re
import codecs
import shutil
import numpy as np
import fnmatch
from textblob import TextBlob
#from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

def get_feature_by_words(filename, word_dict):
    feature = np.zeros((1, len(word_dict.keys())))
    words = filename.strip().split(" ")
    for word in words:
        if word in word_dict.keys():
            feature[0, word_dict[word]] = feature[0, word_dict[word]] + 1

    return feature

def get_words_feature(words):
    global dict_vec
    vecs = []
    print(dict_vec.keys())
    for word in words:
        if word not in dict_vec:
            continue

        vecs.append(dict_vec[word])

    return np.array(vecs)

def get_paper_content(filename):
    with codecs.open(filename, mode="r", encoding="utf-8") as ftxt:
        content = ftxt.readline()

        return content

def get_paper_words(filename):
    content = get_paper_content(filename)
    if len(content) == 0:
        return None

    lem = WordNetLemmatizer()

    content = content[0]
    words = content.split(" ")
    wordsnew = []
    for word in words:
        word = word.lower()
        if check_invalid_word(word):
            continue

        lower_word = word.lower()
        correct_word = TextBlob(lower_word).correct().words[0]
        lemmatize_word = lem.lemmatize(correct_word, "n")

        wordsnew.append(lemmatize_word)

    #print("wordsnew", wordsnew)

    return wordsnew

def get_paper_feature(fnpath):
    words = get_paper_words(fnpath)
    if words is None:
        return None

    feature = get_words_feature(words)
    if len(feature) == 0:
        return None

    normfeature = np.zeros([100, feature.shape[1]])
    validnum = min(100, feature.shape[0])
    normfeature[validnum, :] = feature[validnum, :]

    print(normfeature)
    return normfeature

def getLSIrepresentation(original_matrix, dimension=100):
    matrix_a = original_matrix.toarray()

    u, s, vh = np.linalg.svd(matrix_a)
    s_k = s[:dimension]
    s_k = np.diag(s_k)
    u_k = u[:, :dimension]
    vh_k = vh[:dimension, :]

    return np.matmul(np.matmul(u_k, s_k), vh_k)

def classify_papers(rootdir, paperdir, num_clusters=5, featuretype="content"):
    count_v1 = CountVectorizer(max_df=0.9, min_df=0.02)
    tfidf = TfidfTransformer(norm="l2")

    features = []
    fn_index_dict = {}
    for fn in os.listdir(rootdir):
        fnpath = os.path.join(rootdir, fn)
        if os.path.isdir(fnpath):
            continue

        feature = None
        if featuretype is "name":
            feature = get_normalize_name(fn)
            #if name != None:
            #    features.append(unicode(name))
        elif featuretype is "content":
            feature = get_paper_content(fnpath)
            #if content != None:
            #    features.append(unicode(content))
        else:
            feature = get_paper_feature(fnpath)

        if feature != None:
            fn_index_dict[fn] = len(fn_index_dict)
            features.append(feature)

    if featuretype is "name" or featuretype is "content":
        freq_term_matrix = count_v1.fit_transform(features)

        #tfidf.fit(freq_term_matrix)

        vecs = tfidf.fit_transform(freq_term_matrix)
        vecs = getLSIrepresentation(vecs)
        # word_dict = {}
        # for index, word in enumerate(count_v1.get_feature_names()):
        #     word_dict[word] = index

        # vecs = np.zeros((len(features), len(word_dict.keys())))
        # for index, feature in enumerate(features):
        #     vecs[index] = get_feature_by_words(feature, word_dict)[0]

    else:
        vecs = np.array(features)

    km = KMeans(n_clusters=num_clusters, n_init=100, max_iter=1000).fit(vecs)
    paper_labels = km.labels_

    print(paper_labels)

    for fn in os.listdir(paperdir):
        fnpath = os.path.join(paperdir, fn)
        if os.path.isdir(fnpath):
            continue

        txtfn = fn.replace(".pdf", ".txt")

        txtpath = os.path.join(rootdir, txtfn)

        if not os.path.exists(txtpath):
            continue

        if txtfn not in fn_index_dict.keys():
            dstdir = os.path.join(paperdir, "notsure")
        else:
            predictlabel = paper_labels[fn_index_dict[txtfn]]
            dstdir = os.path.join(paperdir, str(predictlabel))

        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        shutil.copy(os.path.join(paperdir, fn), os.path.join(dstdir, fn))

def init_dict_vec(dictfile):
    global dict_vec
    fkey = open("key.txt", "w")
    with open(dictfile, "r") as fdict:
        word = "nothing"
        while True:

            line = fdict.readline()
            if line is "":
                print(word)
                break

            datas = line.split(" ")
            word = datas[0]

            fkey.write(word + ", ")
            vec = datas[1:]
            vec = map(eval, vec)

            dict_vec[word] = np.array(vec, dtype=np.float16)

    print(dict_vec["and"])
    fkey.close()
    #print(dict_vec.keys())


if __name__=='__main__':
    featuretype = "content"
    if featuretype is None:
        dictfile = r"resource/glove.6B.50d.txt"
        init_dict_vec(dictfile)

    filedir = r"content"
    num_clusters = 20
    paperdir = r"D:\paper"
    classify_papers(filedir, paperdir, num_clusters=num_clusters, featuretype=featuretype)