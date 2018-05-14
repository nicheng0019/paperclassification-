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

def get_words_vector(words):
    global dict_vec

    vecs = []
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
    global dict_vec

    content = get_paper_content(filename)
    if len(content) == 0:
        return None

    #lem = WordNetLemmatizer()

    words = content.split(" ")
    wordsnew = []
    for word in words:
        word = word.lower()
        if word not in dict_vec.keys():
            continue

        lower_word = word.lower()
        if check_invalid_word(lower_word):
            continue

        # correct_word = TextBlob(lower_word).correct().words[0]
        # lemmatize_word = lem.lemmatize(correct_word, "n")

        wordsnew.append(lower_word)

    return wordsnew

def get_paper_feature(fnpath):
    words = get_paper_words(fnpath)
    if words is None:
        return None

    feature = get_words_vector(words)
    if len(feature) == 0:
        return None

    feature = np.mean(feature, axis=0)

    return feature

def getLSIrepresentation(original_matrix, dimension=100):
    matrix_a = original_matrix.toarray()

    u, s, vh = np.linalg.svd(matrix_a)
    s_k = s[:dimension]
    s_k = np.diag(s_k)
    u_k = u[:, :dimension]
    vh_k = vh[:dimension, :]

    return np.matmul(np.matmul(u_k, s_k), vh_k)

def classify_papers(rootdir, paperdir, num_clusters=5, featuretype="tfidf", featureattribute="content"):
    count_v1 = CountVectorizer(max_df=0.9, min_df=0.02)
    tfidf = TfidfTransformer(norm="l2")

    if featuretype is "averagewordvector":
        featureattribute = "vector"

    features = []
    fn_index_dict = {}
    for fn in os.listdir(rootdir):
        fnpath = os.path.join(rootdir, fn)
        if os.path.isdir(fnpath):
            continue

        feature = None
        if featureattribute is "name":
            feature = get_normalize_name(fn)
            #if name != None:
            #    features.append(unicode(name))
        elif featureattribute is "content":
            feature = get_paper_content(fnpath)
            #if content != None:
            #    features.append(unicode(content))
        elif featureattribute is "vector":
            feature = get_paper_feature(fnpath)

        if feature != None:
            fn_index_dict[fn] = len(fn_index_dict)
            features.append(feature)

    if "tfidf" in featuretype:
        freq_term_matrix = count_v1.fit_transform(features)

        #tfidf.fit(freq_term_matrix)

        vecs = tfidf.fit_transform(freq_term_matrix)

        if featuretype is "tfidf+LSI":
            vecs = getLSIrepresentation(vecs)

    elif featuretype is "averagewordvector":
        vecs = np.array(features)

    else:
        print("Value of para featuretype is wrong!", featuretype)
        quit()

    km = KMeans(n_clusters=num_clusters, n_init=100, max_iter=1000).fit(vecs)
    paper_labels = km.labels_

    #print(paper_labels)

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

    with open(dictfile, "r") as fdict:
        word = None
        while True:
            line = fdict.readline()
            if line is "":
                print(word)
                break

            datas = line.split(" ")
            word = datas[0]

            vec = datas[1:]
            vec = map(eval, vec)

            dict_vec[word] = np.array(vec, dtype=np.float16)


if __name__=='__main__':
    featuretype = "averagewordvector"
    featureattribute = "content"
    if featuretype is "averagewordvector":
        dictfile = r"resource/simple50d.txt"
        init_dict_vec(dictfile)

    filedir = r"content"
    num_clusters = 20
    paperdir = r"D:\paper"
    classify_papers(filedir, paperdir, num_clusters=num_clusters, featuretype=featuretype, featureattribute=featureattribute)