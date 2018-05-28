#coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
sess=tf.Session()

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data]) 
y_vals = np.array([x[3] for x in iris.data])
sees = tf.Session()

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

train_indices = np.random.choice(150,120,replace = False)
test_indices = np.array(list(set(range(len(x_vals)))-set(train_indices)))
#将元组转化为列表
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normlize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m-col_min)/(col_max-col_min)

#现在为数据集和目标值声明批量大小的占位符
batch_size = 50
x_data =tf.placeholder(shape = [None,3],dtype = tf.float32)
y_target = tf.placeholder(shape = [None,1],dtype = tf.float32)

#这一步相当重要，声明有合适的形状的模型变量。我们能声明隐藏层为任意大小，本例中有五个节点
hidden_layer_nodes= 5
A1= tf.Variable(tf.random_normal(shape =[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes]))
A2= tf.Variable(tf.random_normal(shape = [hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

#分两部声明训练模型：第一步：创建一个隐藏层输出；第二部：创建训练模型的最后输出
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

#创建损失函数
loss=tf.reduce_mean(tf.square(y_target-final_output))

#声明优化算法，初始化模型变量
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    rand_index = np.random.choice(len(x_vals_train),size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step,feed_dict = {x_data:rand_x,y_target:rand_y})
    if(i+1)%50 ==0:
        print('A1:'+str(sess.run(A1))+'hidden_output:'+str(sess.run(tf.add(tf.matmul(x_data,A1),b1),feed_dict ={x_data:rand_x})))












