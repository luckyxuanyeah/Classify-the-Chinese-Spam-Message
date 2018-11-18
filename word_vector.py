# encoding:utf-8
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
from scipy import sparse, io
# 导入保存训练模型的包
from sklearn.externals import joblib
import re
from sklearn.model_selection import train_test_split
# 用于处理出现非ascii编码的情况，加上reload就可以不报错，处理utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 非 tf-idf 词向量 词袋模型生成每一个文本的特征向量（即统计文本中每行内每个词的词频）
class Counter_Vectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            # 将标点符号去掉
            words = pseg.cut(doc)# pseg进行词性标注
            new_doc = ''.join(w.word for w in words if w.flag != 'x')# 若words中单词w词性非x，将w.word放入new_doc中
            words = jieba.cut(new_doc)# 利用结巴分词将new_doc分成单词，精准模型
            return words# 返回切分好的单词 将词返回给analyzer
        return analyzer# 然后analyzer将词返回给词袋模型 进行词在一行文档中出现频率的统计

# 用tf-idf生成词向量，用这种方法切词速度是非常慢的
# 首先切分单词，修改了继承类
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        # 生成词向量前需要进行切词
        def analyzer(doc):
            # 将标点符号去掉
            words = pseg.cut(doc)# pseg进行词性标注
            new_doc = ''.join(w.word for w in words if w.flag != 'x')# 若words中单词w词性非x，将w.word放入new_doc中
            words = jieba.cut(new_doc)# 利用结巴分词将new_doc分成单词，精准模型
            return words# 返回切分好的单词 将词返回给analyzer
        return analyzer# 然后analyzer将词返回给tfidf模型 统计每个词语的tf-idf权值

# 生成词向量并进行存储，这里是分词模型的训练
def vector_word(content = None):
    if content == None:
        # 读取文件
        with open('RawData/train_content_80w.json', 'r') as f:# 将需要分词的训练数据打开（80万）
            content = json.load(f)# 载入文件
            print 'load train_content finish'

    cut_text_list = []# 新建一个空列表
    count = 0
    all_count = len(content)# 文件的总长度存放于all_count
    pattern = "[\s\.\!\/_,-:;~{}`^\\\[\]<=>?$%^*()+\"\']+|[+——！·【】‘’“”《》，。：；？、~@#￥%……&*（）]+"# 模式字符串
    for text in content:# 对于context中的每一个文本
        count += 1# 计数
        cut_text = jieba.cut(text)# 利用结巴分词将每一行分词
        cut_text = ' '.join(cut_text)# 将分好的词连接成一个以空格为分界的字符串
        # 去标点 re.sub()替换函数 pattern正则表达式中的模式字符串 ' '表示替换成的字符串 将模式字符串替换成空格 后面为要处理的字符串
        text = str(re.sub(pattern.decode('utf-8'),' '.decode('utf-8'),cut_text.decode('utf-8')))# 删cut_text中标点
        cut_text_list.append(text)# 将删除标点后的text词语加入到列表中
        sys.stdout.write('\rcut %d sentences,complete %.2f%%' % (count, (float(count) / all_count) * 100))# all行
        sys.stdout.flush()
    print '\njieba cut finish'

    # 调用了TfidfVectorizer方法初始化向量空间模型 统计每个词语的tf-idf权值  取出频率最高的前2000个词为关键词
    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8,max_features=2000)# 扔频率低，小可能未来出现的词；扔过于频繁词，对匹配相关文档无帮助 （（（（将方法赋值给vec_tfidf））））
    tfidf = vec_tfidf.fit(cut_text_list)# vec_tfidf方法调用删除标点并且分好词语的列表 训练模型 词频矩阵的转化 模型存于tfidf
    joblib.dump(tfidf,'word_vector_model_80w.pkl')# 存储分词模型
    # print 'word_vector_model write finish'
    # data_tfidf = tfidf.transform(cut_text_list)# 得到tfidf的矩阵
    # io.mmwrite('word_vector_80w_1.mtx', data_tfidf)# 将矩阵写入
    # return data_tfidf# 返回矩阵
    # print data_tfidf
    # data_tfidf_dense = data_tfidf.todense()# 将稀疏矩阵转化成完成特征矩阵
    # name_tfidf_feature = vec_tfidf.get_feature_names()


def dispose_new_doc():
    tfidf = joblib.load('word_vector_model.pkl')# 将模型从本地调回
    doc = '您好！紫荆x号本周日x日妇女节有活动，女士到场都有花送，小孩有礼物，下午x:xx还会有抽奖活动哦，有兴趣可过来玩噢！联系人:黄。x'
    transform_document = [doc]# 将文本转化成列表中的第一个元素
    new_data_tfidf = (tfidf.transform(transform_document)).todense()# 使用tfidf模型对这句话进行分词并转成特征矩阵
    print new_data_tfidf

if __name__=='__main__' :
    # vector_word 测试
    # vector_word()
    # print 'word_vector Finish'

    # 嵌入tfidf方法测试
    with open('../RawData/train_content_80w.json','r') as f:# 将内容导入content
        content = json.load(f)
    with open('../RawData/train_label_80w.json','r') as f:# 将标签导入label
        label = json.load(f)
    # 使用随机划分方法将数据划分成训练及和测试集，验证集此处占20%
    train_data, test_data, train_label, test_label = train_test_split(content, label, test_size=0.2, random_state=0)
    tfidf = joblib.load('word_vector_model_80w.pkl')# 将模型从本地调回
    train_data = tfidf.transform(train_data)# 使用tfidf模型对训练数据进行分词并取出关键词
    io.mmwrite('train_data_80w.mtx',train_data)# 将训练数据的tfidf矩阵写入
    test_data = tfidf.transform(test_data)# 使用tfidf模型对测试数据进行分词并取出关键词
    io.mmwrite('test_data_80w.mtx',test_data)# 将测试数据的tfidf矩阵写入

    # dispose_new_doc()
