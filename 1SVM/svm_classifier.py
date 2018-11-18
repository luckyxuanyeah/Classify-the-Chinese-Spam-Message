# encoding:utf-8
import sys,os
# 返回时间戳
import time
# 支持矩阵运算
import numpy as np
# 使用sklearn中的svm算法
from sklearn import svm
# 调用svm的SVC方法
from sklearn.svm import SVC
# 参数集在数据集上的重复使用
from sklearn.pipeline import *
# 使用奇异值数据维度分解将其投影到较低维空间
from sklearn.decomposition import PCA
# 进行网格搜索
from sklearn.model_selection import GridSearchCV
# 进行交叉验证
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
# 用来评估预测误差 计算真实值和预测值之间的误差
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
# 矩阵运算
from scipy import sparse
# 数据分析工具包 数据结构 数学运算 处理缺失数据
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
#从word_vector里面导入TfidfVectorizer方法
from word_vector import TfidfVectorizer

class svm_classifier:
    def __init__(self,content,label):# 载入文件和标签
        self.content = content
        self.train_data,self.test_data,self.train_label,self.test_label = \
            train_test_split(content,label,test_size=0.2,random_state=0)# 将80w分成训练和测试
        # self.clf = svm.svc()
        # 创建模型
        self.model = Pipeline([('clf', SVC())])# 参数clf在数据集上的重复使用

    # 模型初始化
    def model_init(self):
        # 训练集切词
        tf_idf_Vector = TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)
        train_data = tf_idf_Vector.fit_transform(self.train_data)# 将训练数据切词转化成tfidf矩阵
        test_data = tf_idf_Vector.fit_transform(self.test_data)# 测试数据转切词化成tfidf矩阵
        print 'TfidfVectorizer__fit_transform finish'#
        # PCA 特征提取
        # pca = PCA(n_components=1000,copy=True,whiten=True)
        # pca.fit(data_tfidf.todense())
        # train_data_pca = sparse.csr_matrix(pca.transform(self.train_data))
        # test_data_pca = sparse.csr_matrix(pca.transform(self.test_data))
        # print 'PCA finish'

        # return train_data_pca,test_data_pca
        return train_data,test_data# 将训练的矩阵和测试的矩阵返回

    # 调节超参数
    def adjust_paras(self):
        # train_data_pca, test_data_pca = self.model_init()
        train_data,test_data = self.model_init()# 训练的矩阵和测试的矩阵
        parameters_dict = [{'clf__kernel':['linear'],'clf__C':np.logspace(-5,5,11,base=2)},
                           {'clf__kernel': ['rbf'], 'clf__C': np.logspace(-5,5,11,base=2),
                           'clf__gamma':np.logspace(-5,5,11,base=2)}]# 将参数存放在字典中（线性 rbf C gamma）
        # parameters_dict = {'clf__kernel': ['linear'], 'clf__C':[1,2]}

        cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)# 进行五折交叉验证的设计
        print 'shufflesplit finish'
        grid = GridSearchCV(self.model,parameters_dict,cv=cv,n_jobs=4)# 进行网格搜索的设计（估计对象 参数词典 fit的数据 并行运行的作业数，与cpu核数一致，默认1）自动的调节参数
        print 'start fitting'
        # grid.fit(self.train_data,self.train_label)
        grid.fit(train_data,self.train_label)# 利用训练数据调节参数
        print 'fitting finish'

        # 写入超参数
        hyper_paras = pd.DataFrame.from_dict(grid.cv_results_)# 将超参数的结果写入grid.csv中
        with open('hyper_params.csv', 'w') as hyper_paras_f:
            hyper_paras.to_csv(hyper_paras_f)
        print 'hyper_params write finish'

        # 存储模型
        joblib.dump(self.model, 'SVM_model.pkl')# 将SVM分类器模型存储
        print 'model write finish'

        # 打印最优模型参数
        print 'params details'
        best_paras = dict(grid.best_estimator_.get_params())# 从字典中选择最优的参数 grid.best_estimator_方法
        for paras_name in best_paras.keys():
            print '\t%s : %r' % (paras_name,best_paras[paras_name])
        print 'the best score : %.2f' % grid.best_score_

        print "Grid scores on train set:"
        for params, mean_score, scores in grid.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)# 将训练的平均得分训练打印

        # 打印测试信息
        print 'The scores are computed on test set'
        test_result = grid.predict(test_data)# 将测试数据应用于这个参数对应的模型进行预测
        print metrics.classification_report(self.test_label,test_result)# 将准确标签和预测结果比较

    # 定义训练模型的方法
    def model_train(self):
        # train_data, test_data = self.model_init()
        tfidf_vec = TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)# 计算tfidf值
        tfidf = tfidf_vec.fit(self.content)# 使用content来fit模型
        train_data = tfidf.transform(self.train_data)# 用训练数据进行分词并将稀疏矩阵结果转化
        test_data = tfidf.transform(self.test_data)# 用训练数据进行分词并将稀疏矩阵结果转化
        print 'start training SVM'
        start_time = time.time()
        
        #调用SVC方法 分类器代码
        model = svm.SVC(C=8.0,kernel='rbf',gamma=0.25)
        train_model = model.fit(train_data,self.train_label)# 使用最优超参进行训练 训练数据
        print 'model train finish'
        print time.time()-start_time# 输出训练时间
        # 存储模型
        joblib.dump(train_model,'model/SVM_model_train_80w.pkl')# 将训练好的SVM模型存储
        print 'model write finish'
        # 训练误差
        print 'train result'
        train_result = train_model.predict(train_data)# 得到训练的结果
        print metrics.classification_report(self.train_label,train_result)# 将训练结果和正确结果对比并汇报
        # 测试模型
        print 'test result'
        test_result = train_model.predict(test_data)# 用训练好的模型做验证集合的预测
        print metrics.classification_report(self.test_label,test_result)# 将验证集预测结果和正确结果对比

if __name__ == '__main__':
    # 加载数据
    with open('../RawData/train_content_80w.json','r') as f:# 加载文件
        content = json.load(f)
    with open('../RawData/train_label_80w.json','r') as f:# 加载标签
        label = json.load(f)
    classifier = svm_classifier(content,label)# 调用分类器类，传递参数content和label
    # classifier.adjust_paras()
    classifier.model_train()# 进行模型训练，调用训练函数
