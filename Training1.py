import tensorflow as tf

t=[[1,2],[3,4],[5,6],[7,8]]
o=[[1,1],[2,2]]
t1=tf.expand_dims(t,0)
o1=tf.expand_dims(o,1)
sess=tf.Session()
print('t1=', sess.run(t1))
print('o1=', sess.run(o1))

print('subtract=', sess.run(tf.subtract(t1,o1)))
print('square=', sess.run(tf.square(tf.subtract(t1,o1))))