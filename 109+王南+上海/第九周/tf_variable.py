import numpy as np
import tensorflow as tf

counter = tf.Variable(0)
one = tf.constant(1)
add_one = tf.add(counter, one)
update_counter = tf.assign(counter, add_one)

mat = tf.constant(np.arange(15).reshape((3, 5)), dtype=tf.float64)
weight = tf.Variable(np.random.uniform(0, 1, 5).reshape(-1, 1))
result = tf.matmul(mat, weight)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(10):
        session.run(update_counter)
    print(session.run([counter, one, add_one, update_counter]))

    print(session.run([result]))