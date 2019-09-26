
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

def Concatenation(layers):
    return tf.concat(layers, axis=-1)


def conv2d(x, cnum, ksize, stride=1, rate=1, name='conv',
           padding='SAME', activation=tf.nn.relu, training=True):
    assert padding in ['SYMMETRIC', 'VALID', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x



def deconv2d(x, W,stride, cnum):
    x_shape = tf.shape(x)
    weights = tf.Variable(tf.random_normal(W))
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, cnum])
    decon = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')

    return  decon

def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    with tf.variable_scope(name):
        # x = resize(x, func=tf.image.resize_nearest_neighbor)
        W = [3,3,cnum, cnum*2]
        x = deconv2d(x, W, stride = 2, cnum = cnum)
        x = conv2d(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x




def batch_activ_conv(x, out_features, kernel_size, is_training, activation=None, rate=1, name="layer"):
    with tf.variable_scope(name):
        # no dropout!
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
        x = tf.nn.relu(x)
        x = conv2d(x, out_features, kernel_size, activation=activation, rate=rate)
        return x


def Relu(x):
    return tf.nn.relu(x)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def bottleneck_layer(x, scope, training, filters):
    # print(x)
    with tf.name_scope(scope):
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=training, updates_collections=None)
        x = Relu(x)
        x = conv2d(x, 4 * filters, 1, name=scope + '_conv1')
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=training, updates_collections=None)
        x = Relu(x)
        x = conv2d(x, filters, 3, name=scope + '_conv2')
        return x