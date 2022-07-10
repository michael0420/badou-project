import tensorflow as tf

a = tf.constant(5)
b = tf.constant(10)

add = tf.add(a, b)
mul = tf.multiply(a, b)
sub = tf.subtract(b, a)
div = tf.divide(b, a)

with tf.Session() as session:
    # fetch [add, mul, sub, div] result
    print(session.run([add, mul, sub, div]))