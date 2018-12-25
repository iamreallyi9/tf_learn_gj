import tensorflow as tf
import forward
from tensorflow.examples.tutorials.mnist import input_data
import os
#必备常量定义区，知识总结
#batchsize,学习率太大0.1不稳定，太小0.001收敛慢，
#学习率衰减率与基础值相乘
#正则化权重为W的损失的比重
#

BATCH_SIZE = 200
LEARN_RATE_BASE = 0.1
LEARN_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "minst_model"

def backward(mnist):
    #占位x,y_,计算y
    x = tf.placeholder(tf.float32,(None,forward.IN_NODE))
    y_ = tf.placeholder(tf.float32,(None,forward.OUT_NODE))
    y = forward.forward(x,REGULARIZER)


    #以下为优化部分，优化方法包括损失函数-》指数衰减学习率-》滑动平均-》正则化

    #定义loss:使用交叉熵损失函数，步骤固定
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    #指数衰减学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARN_RATE_BASE,
        global_step,
        decay_steps=mnist.train.num_examples/BATCH_SIZE,
        decay_rate=LEARN_RATE_DECAY,
        staircase=True
    )

    #滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    #正则化在定义W时完成

    #模型读取与保存
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #以下三行代码保证了每次训练不会从新训练，从此不再怕断电
        #ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess,ckpt.model_checkpoint_path)


        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            train_value,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print(i,loss_value)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    mnist = input_data.read_data_sets('./m_data',one_hot = True)
    backward(mnist)

if __name__== '__main__':
    main()
