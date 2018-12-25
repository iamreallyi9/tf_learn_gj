import tensorflow as tf

IN_NODE = 784 #输入特征
OUT_NODE = 10 #输出结果
LAYER_NODE = 500 #隐藏层节点

def forward(x,regularizer):
    w1 = get_weight([IN_NODE,LAYER_NODE],regularizer)
    b1 = get_bias(LAYER_NODE)
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

    w2 = get_weight([LAYER_NODE,OUT_NODE],regularizer)
    b2 = get_bias(OUT_NODE)
    y = tf.nn.relu(tf.matmul(y1,w2)+b2)

    return y

def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape,stddev=0.1),dtype=tf.float32)
    if regularizer != None:
        loss = tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
