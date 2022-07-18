import tensorflow as tf

data1 = tf.constant(3.0)
data2 = tf.constant(2.0)
data3 = tf.constant(5.0)

# mul = tf.matmul(data1, data2)  # 将矩阵 a 乘以矩阵 b,生成a * b
mul = tf.add(data2, data3)
ad = tf.multiply(mul, data1)

with tf.Session() as s:
    pro1 = s.run(mul)
    print(pro1)
    pro2 = s.run(ad)
    print(pro2)
    result = s.run([mul, ad])
    print(result)
