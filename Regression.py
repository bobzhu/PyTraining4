import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
num_points=1000
vectors_set=[]
#x_data = np.random.normal(0.0, 0.55,1000)
#y_data = x_data * 01. + 0.3 + np.random.normal(0.0, 0.03)
#print (x_data)
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1*01.+0.3+np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))
y=W*x_data+b
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for step in range(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data,sess.run(W)* x_data+sess.run(b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.legend()
    plt.show()
