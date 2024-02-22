import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2020)

train_df = pd.read_csv('D:/project/datasets/CMAPSSData/train_FD001.txt', sep=" ", header=None)  # train_dr.shape=(20631, 28)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # 去掉26,27列并用新生成的数组替换原数组
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                    's18', 's19', 's20', 's21']
# 先按照'id'列的元素进行排序，当'id'列的元素相同时按照'cycle'列进行排序
train_df = train_df.sort_values(['id', 'cycle'])

test_df = pd.read_csv('D:/project/datasets/CMAPSSData/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                   's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                   's18', 's19', 's20', 's21']

truth_df = pd.read_csv('D:/project/datasets/CMAPSSData/RUL_FD001.txt', sep=" ", header=None)
truth_df2= pd.read_csv('D:/project/datasets/CMAPSSData/RUL_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

"""Data Labeling - generate column RUL"""
# 按照'id'来进行分组，并求出每个组里面'cycle'的最大值,此时它的索引列将变为id
# 所以用reset_index()将索引列还原为最初的索引
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
# 将rul通过'id'合并到train_df上，即在相同'id'时将rul里的max值附在train_df的最后一列
train_df = train_df.merge(rul, on=['id'], how='left')
# 加一列，列名为'RUL'
train_df['RUL'] = train_df['max'] - train_df['cycle']
# 将'max'这一列从train_df中去掉
train_df.drop('max', axis=1, inplace=True)


"""MinMax normalization train"""
# 将'cycle'这一列复制给新的一列'cycle_norm'
train_df['cycle_norm'] = train_df['cycle']
# 在列名里面去掉'id', 'cycle', 'RUL'这三个列名
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])
# 对剩下名字的每一列分别进行特征放缩
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
# 将之前去掉的再加回特征放缩后的列表里面
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
# 恢复原来的索引
train_df = join_df.reindex(columns=train_df.columns)

"""MinMax normalization test"""
# 与上面操作相似，但没有'RUL'这一列
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)


"""generate column max for test data"""
# 第一列是id，第二列是同一个id对应的最大cycle值
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
# 将列名改为id和max
rul.columns = ['id', 'max']
# 给rul文件里的数据列命名为'more'
truth_df.columns = ['more']
# 给truth_df增加id列，值为truth_df的索引加一
truth_df['id'] = truth_df.index + 1
# 给truth_df增加max列，值为rul的max列值加truth_df的more列,
# truth_df['max']的元素是测试集里面每个id的最大cycle值加rul里每个id的真实剩余寿命
truth_df['max'] = rul['max'] + truth_df['more']
# 将'more'这一列从truth_df中去掉
truth_df.drop('more', axis=1, inplace=True)


"""generate RUL for test data"""
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

"""
test_df(13096, 28)

   id  cycle  setting1  setting2  ...       s20       s21  cycle_norm  RUL
0   1      1  0.632184  0.750000  ...  0.558140  0.661834     0.00000  142
1   1      2  0.344828  0.250000  ...  0.682171  0.686827     0.00277  141
2   1      3  0.517241  0.583333  ...  0.728682  0.721348     0.00554  140
3   1      4  0.741379  0.500000  ...  0.666667  0.662110     0.00831  139
...
"""

"""pick a large window size of 50 cycles"""
sequence_length = 50


def gen_sequence(id_df, seq_length, seq_cols):#生成器

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]  #0：50    192-50-1=141：192-1=191 应该有192-50+1=143此循环，这里只有142此 要怎么改？但最后一个窗口可能没用


"""pick the feature columns"""
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)
'''
sequence_cols=['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
'''
# 下一行所用的gen_sequence()中第一个参数是训练集中id为1的部分，第二个参数是50, 第三个参数如下所示
val = list(gen_sequence(train_df[train_df['id'] == 1], sequence_length, sequence_cols))
val_array = np.array(val)  # val_array.shape=(142, 50, 25)  142=192-50

'''
sequence_length= 50
sequence_cols= ['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 
's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_df[train_df['id'] == 1]=

id  cycle  setting1  setting2  ...       s20       s21  RUL  cycle_norm
0     1      1  0.459770  0.166667  ...  0.713178  0.724662  191    0.000000
1     1      2  0.609195  0.250000  ...  0.666667  0.731014  190    0.002770
2     1      3  0.252874  0.750000  ...  0.627907  0.621375  189    0.005540
3     1      4  0.540230  0.500000  ...  0.573643  0.662386  188    0.008310
4     1      5  0.390805  0.333333  ...  0.589147  0.704502  187    0.011080
..   ..    ...       ...       ...  ...       ...       ...  ...         ...
187   1    188  0.114943  0.750000  ...  0.286822  0.089202    4    0.518006
188   1    189  0.465517  0.666667  ...  0.263566  0.301712    3    0.520776
189   1    190  0.344828  0.583333  ...  0.271318  0.239299    2    0.523546
190   1    191  0.500000  0.166667  ...  0.240310  0.324910    1    0.526316
191   1    192  0.551724  0.500000  ...  0.263566  0.097625    0    0.529086

[192 rows x 28 columns]
'''


# region 将每个id对应的训练集转换为一个sequence
seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())     #循环一百次   里面是window number*window size*feature

# 生成sequence并把它转换成np array
# 在train_FD001.txt中按照id分成了100组数据，对每一组进行sequence后每组会减少window_size的大小
# 20631-100*50 = 15631
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)  # seq_array.shape=(15631, 50, 25)
seq_tensor = torch.tensor(seq_array) .to(device)
# seq_tensor = seq_tensor.view(15631, 1, 50, 25).to(device)
print("seq_tensor_shape=", seq_tensor.shape)
print(seq_tensor[0].shape)
#endregion

#region 提取测试数据
def gen_testsequence(id_df, seq_length, seq_cols):

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    if num_elements>=seq_length:
        yield data_array[num_elements-seq_length:, :]
    else:
        yield data_array[:, :]

testseq_gen = (list(gen_testsequence(test_df[test_df['id'] == id], sequence_length, sequence_cols))
           for id in test_df['id'].unique())     #循环一百次   里面是window number*window size*feature

testseq_array = list(testseq_gen)  # seq_array.shape=(15631, 50, 25)
# testseq_tensor = torch.tensor(testseq_array) .to(device)
# seq_tensor = seq_tensor.view(15631, 1, 50, 25).to(device)
# print("seq_tensor_shape=", testseq_tensor.shape)
# print(testseq_tensor[0].shape)
#endregion



"""generate labels"""


def gen_labels(id_df, seq_length, label):

    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
             for id in train_df['id'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)  # label_array.shape=(15631, 1)
label_scale = (label_array-np.min(label_array))/(np.max(label_array)-np.min(label_array))
label_scale=label_array
label_tensor = torch.tensor(label_scale) #直接用ndarrary创建
label_tensor = label_tensor.view(-1)
label_tensor = label_tensor.to(device)
print("label=", label_tensor[:142])


num_sample = len(label_array)
print("num_sample=", num_sample)
input_size = seq_array.shape[2]
hidden_size = 100
num_layers = 2


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.forwardCalculation(x)
        # x = x[-1]
        return x,x[-1]
    # def __init__(self):
    #     super(CNN, self).__init__()
    #     self.conv1 = nn.Sequential(
    #         torch.nn.Conv2d(  # 输入conv1的形状(50, 1, 50, 25)-->输出conv1的形状(50, 20, 26, 13)  为什么是50？
    #             in_channels=1,  # 输入卷积层的图片通道数
    #             out_channels=20,  # 输出的通道数
    #             kernel_size=3,  # 卷积核的大小，长宽相等
    #             stride=1,  # 滑动步长为1
    #             padding=2  # 给输入矩阵周围添两圈0,这样的话在卷积核为3*3时能将输入矩阵的所有元素考虑进去
    #         ),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2) #原文为什么不maxpool呢？
    #     )
    #     self.fc = nn.Linear(20*26*13, 1)  # 将conv1的输出flatten后为(50, 20*26*13)-->经过全连接变为(50, 1)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = x.view(x.size(0), -1)  # 将conv1的输出flatten
    #     # x, _ = self.lstm2(x)
    #     x = self.fc(x)
    #     return x


#region 第二种训练方法，用142的输出，只用50个seq中最后一个的结果来计算loss
##################
lstm2 = LSTM(25, 16, output_size=1, num_layers=1).to(device)
print(lstm2)##########

optimizer = torch.optim.Adam(lstm2.parameters(), lr=0.01)   # optimize all cnn parameters
loss_func = nn.MSELoss()   # the target label is not one-hotted

# loss_sum=0
batch_size=100
for epoch in range(10):
    # for i in range(0, 142):   # 分配 batch data, normalize x when iterate train_loader
    # b_x = seq_tensor[i].view(50, 1, 25)
    i=0
    loss_sum = 0
    while i*batch_size<seq_tensor.size(0):
        if (i+1)*batch_size<seq_tensor.size(0):
            b_x = seq_tensor[i*batch_size:(i+1)*batch_size].transpose(0,1)
            b_y = label_tensor[i*batch_size:(i+1)*batch_size]
        else:
            b_x = seq_tensor[i * batch_size:].transpose(0, 1)
            b_y = label_tensor[i * batch_size:]
        # b_x=b_x.squeeze()
        _,output2 = lstm2(b_x)               # cnn output
        output2=output2.squeeze()
        loss = loss_func(output2, b_y)   # cross entropy loss
        loss_sum+=loss
        # loss_sum+=loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        # output_sum = output
        # print('epoch=%d iteration=%d loss=%f'%(epoch,i, loss))
        # print(loss_sum)
        # loss_sum=0
        i+=1
    print('epoch=%d  total loss=%f'%(epoch, loss))
#endregion

#region test
test_output = np.empty((100,), dtype=np.float32)
for id in test_df['id'].unique():
    b_x=np.array(testseq_array[id-1]).astype(np.float32)
    b_x = torch.tensor(b_x) .transpose(0,1).to(device)
    # b_y=
    print("b_x=", b_x.shape)
    _, output2 = lstm2(b_x)
    test_output[id-1] = output2.item()





#####把100个rul_pred和rul一起画出来
#endregion

# outputplt1 = output1[-1].cpu().detach().numpy()
# plt.plot(outputplt1,label='lstm1')
plt.plot(test_output,label='lstm_test')
plt.plot(truth_df2[0],label='groundtruth')
plt.legend()
plt.show()

# output = lstm(seq_tensor[1].squeeze())
# output=output.unsqueeze(0)
# for i in range(192-50-1):
#     output_temp = lstm(seq_tensor[i+1].squeeze())   # 将第一个sample放进去
#     output=torch.cat((output,output_temp.unsqueeze(0)),0)
# output = output.cpu().detach().numpy()
# label_array = label_tensor[0:192-50].cpu().detach().numpy()
# plt.plot(output)
# plt.plot(label_array)
# plt.show()