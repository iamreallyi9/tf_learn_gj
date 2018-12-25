import tensorflow as tf
import backward
import forward
from PIL import Image
import numpy as np

def model_restore(testpic_arr):
    with tf.Graph().as_default() as tg:

        #描述计算图
        x = tf.placeholder(tf.float32,[None,forward.IN_NODE])
        y = forward.forward(x,None)
        pre_value = tf.arg_max(y,1)

        #滑动平均
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_op = ema.variables_to_restore()
        saver = tf.train.Saver(ema_op)
        #实例化,不是保存//保存和恢复都找saver

        with tf.Session() as sess:
             ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
             if ckpt and ckpt.model_checkpoint_path:
                 saver.restore(sess,ckpt.model_checkpoint_path)
                 pre_value = sess.run(pre_value,feed_dict={x:testpic_arr})
                 return pre_value
             else :
                 print("ERROR@@@")
                 return -1



def pre_pic(testpic):
    #需要PIL库，函数传进来的是普通图片即可
    #喂入网络模型的图片需要满足格式要求：为28*28=784的一维数组，因此会用到resize，以及“灰度化”
    #数值为0到1的浮点数，o代表黑，1代表白，网络训练是为黑底白字，拍照为白底黑字需要处理
    #还可以对像素值进行阈值处理

    img = Image.open(testpic)
    reim = img.resize((28,28),Image.ANTIALIAS)
    #ANTIALIAS为消锯齿整形
    im_arr = np.array(reim.convert('L'))
    #在PIL中，从模式“RGB”转换为“L”模式,即灰度模式，8bit,0-255表示,
    #numpy转化为数组

    threshold = 50 #二值化像素处理的阈值
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 -im_arr[i][j]
            #二值化非黑即白，阈值可调
            if im_arr[i][j] <threshold:
                im_arr[i][j] = 0
            else:im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1,784])

    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0)
    return img_ready

def application():
    test_num = input("你想测试几张图片？")
    for i in range(int(test_num)):
        testpic = input("路径是？")
        #数据预处理
        testpic_arr = pre_pic(testpic)
        #喂入模型
        pre_value = model_restore(testpic_arr)
        print(pre_value)


def main():
    application()

if __name__ == '__main__':
    main()