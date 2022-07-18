import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
input3 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as s:
    print(s.run([output], feed_dict={input1:[7.0],input2:[2.0]}))
