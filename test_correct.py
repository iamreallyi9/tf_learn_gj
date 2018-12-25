import tensorflow as tf
import time
from  tensorflow.examples.tutorials.mnist import input_data
import forward
import backward
import os

TEST_IN_SEC = 5

def test_co(mnist):
    #测试首先要复现模型，也就是需要想，xyy_,w，b【w,b就要考虑滑动平均值】

    #个人理解：会话之前的内容都是描述这个graph的结构，包括输入输出，学习方式，优化方法等，但并不进行实际运算
    #然后在sess里，进行一定次数的运算
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,forward.IN_NODE])
        y_ = tf.placeholder(tf.float32,[None,forward.OUT_NODE])
        y = forward.forward(x,None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
        accuary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #计算正确率，不需要轮数，就是一直在计算，global_step也是一直从别的文件读进来的
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuary_score = sess.run(accuary,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                    print("轮数为 %s 准确率为 %s "%(global_step,accuary_score))
                else:
                    print("error!!!@@@")
                    return
            time.sleep(TEST_IN_SEC)


def main():
    mnist = input_data.read_data_sets("./m_data",one_hot=True)
    test_co(mnist)


if __name__ == '__main__':
    main()
