# -*- coding: utf-8 -*-

import pickle
import numpy as np
import tensorflow as tf

#read data
f=open("D:/data/weibo/weibo_vecs/output/sample/kunmingsample/kmdata_kunmingsample.pkl","rb")
kmdata=pickle.load(f)
f.close()
X_train=kmdata["train"]["data"]
y_train=kmdata["train"]["target"]
X_test=kmdata["test"]["data"]
y_test=kmdata["test"]["target"]
print("data loaded finished")

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
         tf.summary.histogram(name,var)
         mean=tf.reduce_mean(var)
         stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
         tf.summary.scalar('stddev/',+name,stddev)

def weight_variabel(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding="VALID")
def max_pool_2x1(x):
    return tf.nn.max_pool(x,ksize=[1,2,1,1],strides=[1,1,1,1],padding="VALID")
def max_pool_1x2(x):
    return tf.nn.max_pool(x,ksize=[1,1,2,1],strides=[1,1,1,1],padding="VALID")

def one_hot(k,num):
    size=len(k)
    new_k=np.zeros((size,num))
    for idx in range(len(k)):
        new_k[idx][k[idx]]=1
    return new_k


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 100,60])
    y_ = tf.placeholder(tf.float32, shape=[None,2])
    



x_input=tf.reshape(x,[-1,100,60,1])

with tf.name_scope('Convolution_Layer_1'):
    W_conv1=weight_variabel([1,30,1,32])
    b_conv1=bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(x_input,W_conv1)+b_conv1)
with tf.name_scope('Max-pooling_Layer_1'):
    h_pool1=max_pool_2x2(h_conv1)
    #h_pool1_flat=tf.reshape(h_pool1,[-1,99*1*64])

W_conv2=weight_variabel([1,30,32,64])
b_conv2=bias_variable([64])
with tf.name_scope('Convolution_Layer_2'):
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
with tf.name_scope('Max-pooling_Layer_2'):
    h_pool2=max_pool_2x1(h_conv2)
    h_pool2_flat=tf.reshape(h_pool2,[-1,98*1*64])

with tf.name_scope('Full_Connected_Layer_1'):
    W_fc1=weight_variabel([98*1*64,1024])
    b_fc1=bias_variable([1024])



    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

with tf.name_scope('Full_Connected_Layer_2'):
    W_fc2=weight_variabel([1024,2])
    b_fc2=bias_variable([2])

with tf.name_scope('Output_Layer'):
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

with tf.name_scope('Loss_Function'):

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))
    
    tf.summary.scalar('cross_entropy',cross_entropy)


with tf.name_scope('Train_Step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    

    writer=tf.summary.FileWriter("log",tf.get_default_graph())
    writer.close();
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy',accuracy)

init=tf.global_variables_initializer()
merged=tf.summary.merge_all()

saver=tf.train.Saver()
with tf.Session() as sess:

    summary_writer=tf.summary.FileWriter("log",sess.graph)


    sess.run(init)
    batch_size=50
    for i in range(6000):

        #batch=mnist.train.next_batch(50)                    #tain.data
        #sess.run(train_step, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        randnum=np.random.randint(0,len(X_train)-batch_size)
        summary,_=sess.run([merged,train_step], feed_dict={x:X_train[randnum:randnum+batch_size], y_: one_hot(y_train[randnum:randnum+batch_size],2), keep_prob: 0.5})
        summary_writer.add_summary(summary,i)
        if i%200==0:
            acc,ce=sess.run([accuracy,cross_entropy], feed_dict={x:X_train[randnum:randnum+batch_size], y_: one_hot(y_train[randnum:randnum+batch_size],2), keep_prob: 0.5})
            #ce=sess.run(cross_entropy,feed_dict={x:X_train[randnum:randnum+batch_size], y_: one_hot(y_train[randnum:randnum+batch_size],2), keep_prob: 0.5})
            outprint="train accuracy %s cross_entropy %s"%(acc,ce)
            print(outprint)
    
    test_batch_size=1000
    randnum=np.random.randint(0,len(X_test)-batch_size)
    finalacc=sess.run(accuracy,feed_dict={x: X_test[randnum:randnum+test_batch_size], y_: one_hot(y_test[randnum:randnum+test_batch_size],2), keep_prob: 0.5})
    finalprint="test accuracy %s"%finalacc
    print(finalprint)
   
    #saver.save(sess,"D:\\Python\\cnntest\\cnntest\\cnn96save\\20170404\\model.ckpt")
summary_writer.close()
