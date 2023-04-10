import tensorflow as tf

a = tf.Variable([[1.0, 2.0, -0.9],[-0.5, 0.1, 1.2]])
a_nn = tf.nn.relu(a)
sess = tf.Session()
with sess as session:
    tf.global_variables_initializer().run()
    print(sess.run(a_nn))