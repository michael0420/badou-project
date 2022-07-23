import tensorflow as tf
slim = tf.contrib.slim  # 非真实的第三方库，里面有很多模块封装简化代码


def vgg16(inputs, num_classes=1000, is_training=True, dropout_in=0.5,
          spatial_squeeze=True, scope='vgg_16'):
    with tf.variable_scope(scope, [inputs]):  # 命名用的
        # 1、conv2两次[3,3]卷积网络，输出的特征层为64，输出为(224, 224, 64)，再2X2最大池化，输出net为(112, 112, 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # 2、conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出 net为(56,56,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 3、conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net 为(28,28,256）
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # 4、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(56,56,512)，再2X2最大池化，输出net 为(28,28,512）
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 5、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net 为(7,7,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # 6、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)。共进行两次
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_in, is_training=is_training, scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_in, is_training=is_training, scope='dropout7')
        # 7、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        # 8、对卷积后的矩阵平铺，去除维度为1的维度，注意最终为4维，中间2个为1
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
