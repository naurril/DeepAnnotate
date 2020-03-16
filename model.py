import tensorflow as tf
import numpy as np
import math
import sys
import os
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point, num_classes, regression=False):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))  # suffix `pl` means placeholder.
    if regression:
        labels_pl = tf.placeholder(tf.float32, shape=(batch_size))
    else:
        labels_pl =tf.placeholder(tf.int32, shape=(batch_size, len(num_classes)))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_classes, bn_decay=None, regression=False):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    
    #net = tf_util.fully_connected(net, 72, activation_fn=None, scope='fc3')  # so 40 means there are 40 classes.

    # regression
    if regression:
        net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp2')
        
        net = tf_util.fully_connected(net, 16, bn=True, is_training=is_training,
                                  scope='fc4', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp2')
        
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc5')  # so 40 means there are 40 classes.
    else:
        net_final = [x for x in map (lambda i: build_header(net, is_training, bn_decay, num_classes[i], str(i)), range(len(num_classes)))]
        
    return net_final, end_points

def build_header(bone_net, is_training, bn_decay, num_classes, scope_suffix):
    net = bone_net
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1'+scope_suffix, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1'+scope_suffix)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2'+scope_suffix, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2'+scope_suffix)
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3'+scope_suffix)

    return net

def get_loss(pred, label, end_points, class_weight, reg_weight=0.001, regression=False):  #pred is net returned in  `get_model`
    """ pred: B*NUM_CLASSES,
        label: B, """

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    total_loss  =  mat_diff_loss * reg_weight

    if regression:
        #pred = tf.max([pred, 0.0])
        #pred = tf.min([pred, tf.math.pi*2])
        pred = tf.squeeze(pred)
        loss = tf.math.squared_difference(pred, label)
        this_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify loss', this_loss)
        total_loss += this_loss
    else:
        for i in range(len(pred)):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred[i], labels=label[:,i])
            this_loss = tf.reduce_mean(loss) * class_weight[i]
            tf.summary.scalar('classify loss '+str(i), this_loss)
            total_loss += this_loss
    
    return total_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
