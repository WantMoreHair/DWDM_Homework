''' @Author: zhu mengyu  * @Date: 2020-05-01 10:54:47  * @Last Modified by:   zhu mengyu  * @Last Modified time: 2020-05-01 10:54:47  '''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus=["我 来到 北京 清华大学",
        "他 来到 了 网易 杭研 大厦",
        "小明 硕士 毕业 与 中国 科学院",
        "我 爱 北京 天安门"]

# 将文本中的词语转化成词频矩阵
# token_pattern设置未最小词的字数为1
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

# 分词并建立词汇表
vec = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
print("BOW向量")
print(vec.toarray())

# 统计每个单词的tf-idf权值
# norm=None 不进行正则化
tf_idf_transformer = TfidfTransformer(norm=None)

# 将文本转化为词频矩阵并计算tf-idf
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(corpus))

# weight = tf_idf.toarray()
print('tfidf向量')
print(tf_idf.toarray())
