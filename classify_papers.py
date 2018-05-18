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
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

def get_words_vector(words, weights=None):
    global dict_vec

    vecs = []
    for word in words:
        if word not in dict_vec:
            continue

        weight = 1.0
        if weights != None:
            weight = weights[word]

        vecs.append(dict_vec[word] * weight)

    return np.array(vecs)

def get_string_stem(content):
    porter = PorterStemmer()

    new_content = ""
    words = content.split(" ")
    for word in words:
        new_content = new_content + porter.stem(word) + " "

    return new_content

def get_paper_content(filename):
    with codecs.open(filename, mode="r", encoding="utf-8") as ftxt:
        content = ftxt.readline().strip()

        return get_string_stem(content)#[porter.stem(word) for word in content.split(" ")]

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

def get_paper_vector(fnpath, weights=None):
    words = get_paper_words(fnpath)
    if words is None:
        return None

    feature = get_words_vector(words, weights=weights)
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
        elif "content" in featureattribute:
            feature = get_paper_content(fnpath)
        elif featureattribute is "vector":
            feature = get_paper_vector(fnpath)

        if feature != None:
            fn_index_dict[fn] = len(fn_index_dict)
            features.append(feature)

    if "tfidf" in featuretype:
        count_v1 = CountVectorizer(max_df=0.9, min_df=0.02, stop_words=stopwords.words('english'))
        freq_term_matrix = count_v1.fit_transform(features)

        vecs = tfidf.fit_transform(freq_term_matrix)
        if featuretype is "tfidf+LSI":
            vecs = getLSIrepresentation(vecs)

    elif featuretype is "sif":
        count_v1 = CountVectorizer()
        freq_term_matrix = count_v1.fit_transform(features)
        freq_word_array = np.sum(freq_term_matrix, axis=0, dtype=np.float32)
        freq_word_array = freq_word_array / np.sum(freq_word_array, dtype=np.float32)

        weight_dict = {}
        a = 0.0001
        for word in count_v1.vocabulary_.keys():
            weight_dict[word] = a / (a + freq_word_array[0, count_v1.vocabulary_[word]])

        vecs = []
        fn_index_dict = {}
        for fn in os.listdir(rootdir):
            if os.path.isdir(fnpath):
                continue

            vec = get_paper_vector(fnpath, weights=weight_dict)
            if vec != None:
                fn_index_dict[fn] = len(fn_index_dict)
                vecs.append(vec)
        vecs = np.array(vecs, dtype=np.float32)

        u, s, vh = np.linalg.svd(vecs)
        s[0] = 0
        s = np.diag(s)
        n1 = u.shape[0]
        n2 = vh.shape[0]
        n3 = s.shape[0]
        news = np.zeros((n1, n2))
        news[:n3, :n3] = s
        vecs = np.matmul(np.matmul(u, news), vh)

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
        while True:
            line = fdict.readline()
            if line is "":
                break

            datas = line.split(" ")
            word = datas[0]

            vec = datas[1:]
            vec = map(eval, vec)

            dict_vec[word] = np.array(vec, dtype=np.float16)


if __name__=='__main__':
    featuretype = "sif"
    featureattribute = "vector+content"
    if "vector" in featureattribute:
        dictfile = r"resource/simple50d.txt"
        init_dict_vec(dictfile)

    filedir = r"content"
    num_clusters = 20
    paperdir = r"D:\paper"
    classify_papers(filedir, paperdir, num_clusters=num_clusters, featuretype=featuretype, featureattribute=featureattribute)