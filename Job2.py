import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def comput_y(x):
    return 2.5 * x * np.cos(5*x) + 5.8 * x +10 +np.random.randn(x.shape[0])

def w_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev=0.1)
    return tf.Variable(initial)

def b_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

x_t = np.linspace(-2,2,2000).reshape([1,-1])
print(x_t.shape)

y_t = comput_y(x_t)


x = tf.placeholder(tf.float32, [1,2000])
hiddennum = 150

w = w_variable([hiddennum,1])
b = b_variable([hiddennum,1])

w2 = w_variable([1,hiddennum])
b2 = b_variable([1])

hidden = tf.nn.sigmoid(tf.matmul(w,x)+b)
y = tf.matmul(w2,hidden)+b2

loss = tf.reduce_mean(tf.square(y-y_t))
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(0.1,step,10,0.99,staircase=True)
optimizer = tf.train.AdamOptimizer(rate)
train = optimizer.minimize(loss, global_step=step)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for time in range(0,20001):
    train.run({x:x_t},sess)
    if time%200 == 0:
        print("训练次数：", time, "Loss = ", loss.eval({x:x_t},sess),
              "rate = ", rate.eval({x:x_t},sess),
              "Compute_rate = ", 0.1*math.pow(0.99,(1+time)//10))

plt.plot(x_t[0], y_t[0], 'mo', label = 'test data')
plt.plot(x_t[0], y.eval({x:x_t},sess)[0], label = "comput data")
plt.legend()
plt.show()







