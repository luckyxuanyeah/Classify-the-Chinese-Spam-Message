# encoding:utf-8
import time
from sklearn.externals import joblib
import numpy as np
import xgboost as xgb
from sklearn import cross_validation,metrics
import pandas as pd
import matplotlib.pylab as plt
from xgboost.sklearn import XGBClassifier
from scipy import io
import json
from sklearn.model_selection import train_test_split
import os, sys
from sklearn.model_selection import GridSearchCV
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from word_vector import TfidfVectorizer


# 使用xgboost默认参数
def model_default(train_data,train_label,test_data,test_label):
    params = {'silent':0,'nthread':4,'objective':'multi:softmax','num_class':2,'seed':0}
    dtrain_data = xgb.DMatrix(data=train_data,label=train_label)
    dtest_data = xgb.DMatrix(data=test_data,label=test_label)
    eval_list = [(dtrain_data,'train'),(dtest_data,'test')]
    xgb_train_default = xgb.train(params=params,dtrain=dtrain_data,num_boost_round=50,evals=eval_list)
    # 存储模型
    xgb_train_default.save_model('model/xgboost_5k.model')
    xgb_train_default.dump_model('dump.raw.txt')
    dtest_data = xgb.DMatrix(test_data)
    preds = xgb_train_default.predict(dtest_data)
    print preds[0]
    print metrics.accuracy_score(test_label,preds)
    print 'ok'
def modelfit(model,train_data,train_label,useTrain_cv = True,cv_folds=5,early_stopping_rounds=50):
    if useTrain_cv:
        xgb_params = model.get_xgb_params()
        xgb_data = xgb.DMatrix(data=train_data,label=train_label)
        cv_result = xgb.cv(params=xgb_params,dtrain=xgb_data,num_boost_round=model.get_params()['n_estimators'],
                           nfold=cv_folds,metrics=['auc'],early_stopping_rounds=early_stopping_rounds)
        print cv_result
        model.set_params(n_estimators=cv_result.shape[0])
    # 训练数据
    model.fit(train_data,train_label,eval_metric='auc')
    # 测试训练结果
    train_predictions = model.predict(train_data)
    train_predprob = model.predict_proba(train_data)[:,1]
    # 打印训练结果
    print 'Model report:'
    print 'Accuracy : %.4g' % metrics.accuracy_score(train_label,train_predictions)
    # print 'AUC Score(Train): %f' % metrics.roc_auc_score(train_label,train_predprob)
    b=model.get_params()
    a=model.get_booster()
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    print 'ok'


'''
1.利用网格搜索确定 n_estimators参数    result : n_estimator=160
2.网格搜索 max_depth min_child_weight 参数 result :max_depth=6,min_child_weight=3
'''
def choose_params(train_data,train_label,test_data,test_label):
    # params_dict = {'n_estimators':np.arange(60,200,10)}
    params_dict = {'max_depth':np.arange(2,10,1),'min_child_weight':np.arange(1,6,1)}
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=160,
                          max_depth=6,
                          min_child_weight=1,
                          gamma=0,subsample=0.8,colsample_bytree=0.8,
                          objective ='multi:softmax', num_class=2,
                          nthread=4,scale_pos_weight=1,seed=0)
    # modelfit(model,train_data,train_label)
    print 'start grid search'
    grid = GridSearchCV(model,params_dict,n_jobs=4,cv=5)
    print 'grid search finish'
    print 'start fit model'
    grid.fit(train_data,train_label)
    print 'fit model finish'
    # 写入超参数
    hyper_paras = pd.DataFrame.from_dict(grid.cv_results_)
    with open('hyper_params_depth_child_weight.csv', 'w') as hyper_paras_f:
        hyper_paras.to_csv(hyper_paras_f)
    print 'hyper_params write finish'

    print "Grid scores on train set:"
    for params, mean_score, scores in grid.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)

    # 打印测试信息
    print 'The scores are computed on test set'
    test_result = grid.predict(test_data)
    print metrics.classification_report(test_label, test_result)

# 训练最终模型
def train_model(train_data,train_label,test_data,test_label):
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=160,
                          max_depth=6,
                          min_child_weight=3,
                          gamma=0,subsample=0.8,colsample_bytree=0.8,
                          objective ='multi:softmax', num_class=2,
                          nthread=4,scale_pos_weight=1,seed=0)
    # dtrain = xgb.DMatrix(data=train_data,label=train_label)
    # 训练模型
    print 'XGboost start trainning '
    start_time = time.time()
    model.fit(train_data,train_label)
    print 'XGboost finish trainning'
    print 'trainning time:%d' % (time.time() - start_time)
    # 存储模型
    joblib.dump(model,'model/XGBoost_model_80w.pkl')
    print 'model write finish'
    # 测试结果
    print 'train data result'
    train_result = model.predict(train_data)
    print metrics.classification_report(train_label,train_result)
    print 'test data result'
    test_result = model.predict(test_data)
    print metrics.classification_report(test_label,test_result)


if __name__ == '__main__':

    with open('../RawData/train_content_80w.json','r') as f:
        content = json.load(f)
    with open('../RawData/train_label_80w.json','r') as f:
        label = json.load(f)
    train_data,test_data,train_label,test_label = train_test_split(content,label,test_size=0.2,random_state=0)
    vec_tfidf = TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)
    tfidf = vec_tfidf.fit(content)
    # 存储 tfidf 模型
    joblib.dump(tfidf, 'model/word_vector_model_80w.pkl')
    print 'word_vector_model write finish'
    # 读取tfidf模型
    # tfidf = joblib.load('model/word_vector_model_60w.pkl')

    train_data = tfidf.transform(train_data)
    test_data = tfidf.transform(test_data)
    # train_data = np.array(train_data.todense())
    # test_data = np.array(test_data.todense())

    # 原始参数
    # model_default(train_data,np.array(train_label),test_data,np.array(test_label))

    # 网格搜索超参数
    # choose_params(train_data,np.array(train_label),test_data,np.array(test_label))

    # 使用最优超参数训练模型
    train_model(train_data,np.array(train_label),test_data,np.array(test_label))