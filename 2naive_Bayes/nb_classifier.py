# encoding:utf-8
import json
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from word_vector import TfidfVectorizer

def train():
    # 准备数据
    with open('../RawData/train_content_80w.json','r') as f:
        content = json.load(f)
        print 'train_content load finish'
    with open('../RawData/train_label_80w.json','r') as f:
        label = json.load(f)
    print 'train_label load finish'
    # 随机切分数据
    train_data,test_data,train_label,test_label = train_test_split(content,label,test_size=0.2,random_state=0)
    # 载入模型
    # tfidf = joblib.load('model/word_vector_model_80w.pkl')
    tfidf_vec = TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)
    tfidf = tfidf_vec.fit(content)
    train_data = tfidf.transform(train_data)
    test_data = tfidf.transform(test_data)

    # 建立多项式naiveBayes模型
    clf = MultinomialNB()
    print 'start trainning model'
    model = clf.fit(train_data,train_label)
    print 'model trainning finish'
    # 存储模型
    joblib.dump(model, 'model/Multi_NB_model_80w.pkl')
    print 'model write finish'
    # 测试结果
    print 'train data result'
    train_result = model.predict(train_data)
    print metrics.classification_report(train_label, train_result)
    print 'test data result'
    test_result = model.predict(test_data)
    print metrics.classification_report(test_label, test_result)

if __name__ == '__main__':
    train()
