# -*- coding:utf8 -*-
import os
import re
import codecs
import shutil
import numpy as np
from textblob import TextBlob
#from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer

def getnormalizename(originalname):
    lem = WordNetLemmatizer()

    pattern = '_[0-9]+.[0-9]+(.*)pdf'
    searchobj = re.search(pattern, originalname)
    result = ".pdf"
    if searchobj != None:
        result = searchobj.group()

    name = originalname.replace(result, "")

    name = name.strip()
    name = name.replace(",", "")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("[", "")
    name = name.replace("]", "")
    name = name.replace("+", "")
    name = name.replace("-", " ")
    name = name.replace("    ", " ")
    name = name.replace("   ", " ")
    name = name.replace("  ", " ")
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

def getfeaturebyfilename(filename, word_dict):
    feature = np.zeros((1, len(word_dict.keys())))
    words = filename.strip().split(" ")
    for word in words:
        if word in word_dict.keys():
            feature[0, word_dict[word]] = feature[0, word_dict[word]] + 1

    return feature

def classifypaperfiles(paperdir, namelistfile, num_clusters=5):
    ftxt = codecs.open(namelistfile, encoding="utf-8", mode="w")

    rootdir = paperdir
    paper_names = []
    for fn in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, fn)):
            continue

        normal_name = getnormalizename(fn)

        if normal_name != None:
            paper_names.append(unicode(normal_name))
            ftxt.write(normal_name)
            ftxt.write("\r\n")

    ftxt.close()

    count_v1 = CountVectorizer(max_df=0.8, min_df=0.01)
    counts_train = count_v1.fit_transform(paper_names)

    word_dict = {}
    for index, word in enumerate(count_v1.get_feature_names()):
        word_dict[word] = index

    features = np.zeros((len(paper_names), len(word_dict.keys())))
    for index, name in enumerate(paper_names):
        features[index] = getfeaturebyfilename(name, word_dict)[0]

    km = KMeans(n_clusters=num_clusters).fit(features)
    paper_labels = km.labels_

    print(paper_labels)

    for fn in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, fn)):
            continue

        normal_name = getnormalizename(fn)

        if normal_name != None:
            feature = getfeaturebyfilename(normal_name, word_dict)
            predictresult = km.predict(feature)
            dstdir = os.path.join(rootdir, str(predictresult[0]))
        else:
            dstdir = os.path.join(rootdir, "notsure")

        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        shutil.copy(os.path.join(rootdir, fn), os.path.join(dstdir, fn))

if __name__=='__main__':
    paperdir = r"D:\paper"
    namelistfile = "pdf_train.txt"
    num_clusters = 10
    classifypaperfiles(paperdir, namelistfile, num_clusters=num_clusters)
