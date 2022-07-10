import tensorflow as tf
import numpy as np

# the placeholers for feed
a = tf.placeholder(dtype=tf.float32, shape=(4, 3), name="x")
b = tf.placeholder(dtype=tf.float32, shape=(3, 5), name="y")

xy = tf.matmul(a, b)

with tf.Session() as session:
    x0 = np.linspace(0, 1, 12).reshape((4, 3))
    y0 = np.linspace(0, 1, 15).reshape((3, 5))
    # feed the placeholders
    result = session.run(xy, feed_dict={a: x0, b: y0})
    print(result)