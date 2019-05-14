import tensorflow as tf
import numpy as np

#X = tf.placeholder("float")
#Y = tf.placeholder("float")
#W = tf.Variable(np.random.random(), name="weight")
#pred = tf.multiply(X, W)
#cost = tf.reduce_sum(tf.pow(pred-Y, 2))
#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    for t in range(10000):
#        x = np.array(np.random.random()).reshape((1, 1, 1, 1))
#        y = x * 3
#        (_, c) = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
#        print(c)

c = np.random.random([4, 2])
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('c:',c)
    print(sess.run(b))

w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    ema_val = tf.identity(ema.average(update))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        print(sess.run([ema_val]))