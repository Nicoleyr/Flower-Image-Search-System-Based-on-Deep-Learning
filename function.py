import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.1
decay_rate = 0.96
global_steps = 1000
decay_steps = 100

# tf.Variable:主要在于一些可训练变量（Trainable variables）, 比如权值和偏置，声明时必须提供初始值，在训练时其值会发生改变。
# tf.placeholder：用于得到传递进来的真实训练样本，不必指定初始值，可在run时通过feed_dict参数指定

# global_ = tf.Variable(0)
global_ = tf.placeholder(dtype=tf.int32)
print(type(global_))
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_steps), F_D, 'r-')
plt.plot(range(global_steps), T_C, 'b-')

plt.show()



