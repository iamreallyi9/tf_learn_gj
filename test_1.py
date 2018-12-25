import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('./m_data',one_hot = True)

#打印数据集信息
#print (minst.train.num_examples)
#print(minst.validation.num_examples)
#print(minst.test.num_examples)
#print(minst.train.labels[0])
#print(minst.train.images[0])

BATCH_SIZE = 200
#随机抽取出BATCH_SIZE个信息进行训练
xs, ys = minst.train.next_batch(BATCH_SIZE)
print(xs,ys.shape)


