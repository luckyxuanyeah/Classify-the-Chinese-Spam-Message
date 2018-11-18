# encoding:utf-8
from sklearn.externals import joblib
import sys,os
import sklearn.feature_extraction.text
# from word_vector import Counter_Vectorizer,TfidfVectorizer
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
#从word_vector里面导入TfidfVectorizer方法
from word_vector import TfidfVectorizer

#预测时将两个模型加载进来 分词模型和分类模型
class svm_predict():
    def __init__(self,classifier_model,word_vector_model):# 初始化参数
        self.classifier_model = classifier_model# 加载分类模型
        self.word_vector_model = word_vector_model# 加载分词模型

    def test_data_predict(self,doc):
        count = 0
        test_dataFile_len = len(doc)# 求出doc的行数
        for doc_line in doc:# 在doc中的每一行循环操作
            doc_tfidf = self.word_vector_model.transform([doc_line])# 将doc中的行进行分词并作稀疏矩阵转化
            predict_line = self.classifier_model.predict(doc_tfidf)# 将矩阵输入进行预测结果
            with open('predict_result/svm_predict.txt','a') as f:
                f.writelines(predict_line[0]+'\t'+doc_line)# 将结果加入到文件中
            count += 1# 进行计数加1
            sys.stdout.write('\rpredict %d message,complete %.2f%%' % (count,(float(count)/test_dataFile_len)*100))# 查看已经处理了多少条数据
            sys.stdout.flush()

        # doc_tfidf = self.word_vector_model.transform(doc)
        # predict_result = self.classifier_model.predict(doc_tfidf)
        # with open('predict_result/xgb_predict.txt','w') as f:
        #     for i in range(len(predict_result)):
        #         f.writelines(predict_result[i])
        # print 'XGBoost predict_result write finish '
    # 一条的处理
    def single_sample_predict(self,doc):
        doc = [doc]
        single_sample_tfidf = self.word_vector_model.transform(doc)
        predict_result = self.classifier_model.predict(single_sample_tfidf)
        print predict_result
    # 预测方法 选择模式（简单预测 预测文件）
    def predict(self,operate_methods,single_sample=None,docment=None):
        if operate_methods == 'single':
            if single_sample == None:
                print 'You hava to provide single sample.No sample has been given.'
            else:
                self.single_sample_predict(single_sample)
        if operate_methods == 'set':
            if docment == None:
                print 'You hava to provide document set.No document set has given.'
            else:
                self.test_data_predict(docment)

if __name__ == '__main__':
    classifier_model = joblib.load('model/SVM_model_train_60w.pkl')# 从本地加载SVM分类模型
    print 'classifier_model load finish'
    word_vector_model = joblib.load('model/word_vector_model_60w.pkl')# 从本地加载分词模型
    print 'word_vector_model load finish'
    predict = svm_predict(classifier_model,word_vector_model)# 调用svm_predict方法 参数为两个模型

    # 在线测试样本集合
    with open('../test_online.txt') as fr:# 将数据加载进来
        doc = fr.readlines()
    predict.predict('set',docment=doc)
    '''
    # 在线测试单个输入样本 传入一段话后可以判断是否为垃圾文件
    while(True):
        doc = raw_input()
        if doc == 'exit':
            break
        predict.predict('single',doc)
    '''
