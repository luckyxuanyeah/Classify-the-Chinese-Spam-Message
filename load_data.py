# encoding:utf-8
import json

# 加载原始数据，进行content/label分割
def load_message():
    content = [] #定义空列表
    label = []
    lines =[]

    # 打开原数据文件
    with open('data.txt') as fr:
        data_size = 0 #定义一个存放行数的值
        line = fr.readline() #将文件中的一行读入存放line
        while line: #当line有值时
            lines.append(line) #将这一行加入到空列表lines中
            data_size += 1 #将data_size值加一
            if data_size == 800000: #如果data_size的值达到了80000个
                break #退出循环
            line = fr.readline() #如果还没有到80000，读出下一行赋值给line，line还有值则继续循环
        # 记录数据规模
        num = len(lines) #lines中存放了所有的行，num为行数
        for i in range(num): #从0-（num-1）循环
            message = lines[i].split('\t') #将循环到的这行进行分割，得到信息文本和label，存于message
            label.append(message[0]) #将message[0]放入label列表中
            content.append(message[1]) #将message[1]放入content列表中
    return num, content, label #返回数量num，列表content和label

# 将分割后的原始数据存到json
def data_storage(content, label):
    with open('RawData/train_content_80w.json', 'w') as f: #将content存到json中
        json.dump(content, f) #dump是将dict（字典）数据转化成json数据后写入json文件
    with open('RawData/train_label_80w.json', 'w') as f: #将label存到json中
        json.dump(label, f)

#相当于主函数，调用了两个方法
if '__main__' == __name__:
    num, content, label = load_message() #load_message有三个返回值赋给三个num，content，label
    data_storage(content, label) #运行data_storage
