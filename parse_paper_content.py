# -*- coding:utf8 -*-
import os
from HTMLParser import HTMLParser

from utils import *

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    content = ""
    def handle_starttag(self, tag, attrs):
        #print "Encountered a start tag:", tag
        pass

    def handle_endtag(self, tag):
        #print "Encountered an end tag :", tag
        pass

    def handle_data(self, data):
        #print "Encountered some data  :", data
        self.content = self.content + " " + data

def parse_paper_content(fname, dstdir='content'):
    # instantiate the parser and fed it some HTML
    parser = MyHTMLParser()

    with open(fname, "rb") as fhtml:
        contents = fhtml.readlines()

    for line in contents:
        parser.feed(line)

    data = clean_string(parser.content)

    txtname = os.path.basename(fname).replace(".html", ".txt")
    with open(os.path.join(dstdir, txtname), "w") as ftxt:
        ftxt.write(data)

if __name__ == '__main__':
    import fnmatch
    htmldir = r"data_p5/"
    dstdir = "content"
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)

    for fname in os.listdir(htmldir):
        if fnmatch.fnmatch(fname, "*.html"):
            parse_paper_content(os.path.join(htmldir, fname), dstdir)