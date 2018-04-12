# -*- coding:utf8 -*-
import os
import re
import codecs
import shutil

def getpdfname(pdfdir, namelistfile):
    notsuredir = r"notsure"
    if not os.path.exists(notsuredir):
        os.makedirs(notsuredir)

    word_dict = []
    ftxt = codecs.open(namelistfile, encoding="utf-8", mode="w")
    rootdir = pdfdir
    for fn in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, fn)):
            continue

        pattern = '_[0-9]+.[0-9]+(.*)pdf'
        searchobj = re.search(pattern, fn)
        result = ".pdf"
        if searchobj != None:
            result = searchobj.group()
            print(searchobj.group())
        name = fn.replace(result, "")

        words = re.split(r"[;._\s]", name)
        try:
            for word in words:
                ftxt.write(word.lower() + " ")
            ftxt.write("\r\n")
        except:
            print("file name: ", words)
            shutil.copy(os.path.join(rootdir, fn), os.path.join(notsuredir, fn))

    ftxt.close()

if __name__=='__main__':
    pdfdir = r"D:\paper"
    namelistfile = "pdf_train.txt"
    getpdfname(pdfdir, namelistfile)
