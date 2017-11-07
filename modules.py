from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim


class ModuleC(object):
    def __init__(self, config):
        self.config = config
        self.network = importlib.import_module('facenet.src.'+config.model_def)


    def getNetwork(self,image, nrof_classes, label_batch, reuse= False):
        # Build the inference graph
        prelogits, _ = self.network.inference(image, self.config.keep_probability, phase_train = False, bottleneck_layer_size=self.config.embedding_size, weight_decay=self.config.weight_decay, reuse=reuse)
        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                      scope='Logits', reuse=reuse)

        #embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.facenet_scope)
        logit_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Logits')

        return total_loss, variables, logit_variables


    def getFirstConv(self,image,reuse=None):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],normalizer_fn=slim.batch_norm):
            with tf.variable_scope('InceptionResnetV1', 'InceptionResnetV1', [image], reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                        stride=1, padding='SAME'):
                        # 149 x 149 x 32
                        net = slim.conv2d(image, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        # 147 x 147 x 32
                        net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.facenet_scope)

        return net, variables


