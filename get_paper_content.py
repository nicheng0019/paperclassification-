#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter
import os

def get_paper_content(fname):
    debug = 0
    # input option
    password = ''
    pagenos = set()
    maxpages = 0
    # output option
    basename = os.path.basename(fname)
    basename = basename.replace(".pdf", "")
    outfile = os.path.join("data", basename + ".html")
    outtype = None
    imagewriter = None
    rotation = 0
    stripcontrol = False
    layoutmode = 'normal'
    codec = 'utf-8'
    pageno = 1
    scale = 1
    caching = True
    showpageno = True
    laparams = LAParams()

    PDFDocument.debug = debug
    PDFParser.debug = debug
    CMapDB.debug = debug
    PDFPageInterpreter.debug = debug
    #
    rsrcmgr = PDFResourceManager(caching=caching)
    if not outtype:
        outtype = 'text'
        if outfile:
            if outfile.endswith('.htm') or outfile.endswith('.html'):
                outtype = 'html'
            elif outfile.endswith('.xml'):
                outtype = 'xml'
            elif outfile.endswith('.tag'):
                outtype = 'tag'
    if outfile:
        outfp = file(outfile, 'w')
    else:
        outfp = sys.stdout
    if outtype == 'text':
        device = TextConverter(rsrcmgr, outfp, codec=codec, laparams=laparams,
                               imagewriter=imagewriter)
    elif outtype == 'xml':
        device = XMLConverter(rsrcmgr, outfp, codec=codec, laparams=laparams,
                              imagewriter=imagewriter,
                              stripcontrol=stripcontrol)
    elif outtype == 'html':
        device = HTMLConverter(rsrcmgr, outfp, codec=codec, scale=scale,
                               layoutmode=layoutmode, laparams=laparams,
                               imagewriter=imagewriter, debug=debug)
    elif outtype == 'tag':
        device = TagExtractor(rsrcmgr, outfp, codec=codec)
    else:
        return

    fp = file(fname, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    interpreter.debug = True
    try:
        for index, page in enumerate(PDFPage.get_pages(fp, pagenos,
                                      maxpages=maxpages, password=password,
                                      caching=caching, check_extractable=True)):
            if index > 2:
                break
            page.rotate = (page.rotate+rotation) % 360
            interpreter.process_page(page)
    except:
        print(fname)
        return

    fp.close()
    device.close()
    outfp.close()
    return

if __name__ == '__main__':
    import fnmatch
    fdir = r"D:\paper"
    for fname in os.listdir(fdir):
        if fnmatch.fnmatch(fname, "*.pdf"):
            #print(fname)
            get_paper_content(os.path.join(fdir, fname))
