# paperclassification-
Unsupervised classification by the topics of papers

20180412 工程创建：

    下载的论文已经有300多篇了，大部分都没有分类，现在已经忘了都是关于什么内容的了，找起来也很麻烦。于是打算根据论文题目做一个粗略的分类。

可能会涉及到的处理：

    翻译（多数论文是英文名字，个别中文名字的翻译成英文）
    
    纠错（论文名字中有单词拼写错误）
    
    分类（具体用哪种分类方法还没想出来）


20180508 算法改进：
    
    最初版本使用bag-of-words方法作为特征，又改为使用tf-idf特征，效果有提高，但是会出现很多文章聚集在其中一类的情况，需要再进一步细分。
    
    在得到tf-idf之后，再做SVD分解（Latent Semantic Indexing，LSI，参考论文《Latent Semantic Indexing: An overview》），效果又有提高。
    
    
20180514 算法尝试：
  
  把文本里的words的vectors取均值作为文本的vector，再做聚类，分类的效果不好，可能是因为文本的vector距离近的，不代表文本的topic相近，所以通常是在监督学习的情况下使用这种方法。
