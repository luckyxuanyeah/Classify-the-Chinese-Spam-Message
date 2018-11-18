# Classify-the-Chinese-Spam-Message
这个项目旨在进行中文垃圾短信的分类处理。在这个信息化时代，我们的手机经常会接收到五花八门的短信，对于用户来说，一个高精度的自动检测垃圾短信的程序是十分具有使用价值的。

项目简介：
</br>load_data.py:txt文本数据的载入以及数据content、label切分
</br>word_vector.py:中文文本数据的切词以及TF-IDF的计算
</br>naive_bayes、SVM、XGBoost三个文件夹中分别是基于当前模型的分类器的实现:
</br>model:存放训练好的模型
</br>predict:无标签短信文本的预测结果
</br>word_vector:词向量矩阵的存储
</br>以及实验调参的过程(.csv文件)和实验结果截图
