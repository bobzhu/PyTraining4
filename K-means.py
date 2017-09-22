import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
num_puntos=4
conjunto_puntos=[]
for i in range(num_puntos):
    if np.random.random()>0.5:
        conjunto_puntos.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df=pd.DataFrame({"x":[ v[0] for v in conjunto_puntos],"y":[v[1] for v in conjunto_puntos]})
sns.lmplot("x","y",data=df, fit_reg=False, size=6)
#plt.show()

sess = tf.Session()

vectors = tf.constant(conjunto_puntos)  # 2000个向量
k = 2
centroides =tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]) # 重新排列向量(random_shuffle)，并取前面四个作为初始质心
expanded_vectors = tf.expand_dims(vectors, 0)  # 在最前面增加一个维度，成为1*2000二维矩阵
expanded_centroides = tf.expand_dims(centroides, 1)  # 增加一个维度，成为4*1二维矩阵
#assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)  #求每个点和质心距离的平方
print(expanded_vectors, expanded_centroides)
print('expanded_vectors=', sess.run(expanded_vectors))
print('expanded_centroides=', sess.run(expanded_centroides))
print('subtract=', sess.run(tf.subtract(expanded_vectors, expanded_centroides)))
print('square=', sess.run(tf.square(tf.subtract(expanded_vectors, expanded_centroides))))
#print('assignments=', sess.run(assignments))

#print(sess.run(tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1]))) for c in range(k)))

#means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in range(k)])
# update_centroides = tf.assign(centroides, means)
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# for step in range(100):
#    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
# data = {"x": [], "y": [], "cluster": []}
# for i in range(len(assignment_values)):
#   data["x"].append(conjunto_puntos[i][0])
#   data["y"].append(conjunto_puntos[i][1])
#   data["cluster"].append(assignment_values[i])
# df = pd.DataFrame(data)
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
# plt.show()