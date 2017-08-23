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


    def getNetwork(self,image, nrof_classes, label_batch):
        # Build the inference graph
        prelogits, _ = self.network.inference(image, self.config.keep_probability, bottleneck_layer_size=self.config.embedding_size,
                                         weight_decay=self.config.weight_decay)
        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

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

        sess = tf.Session()
        saver = tf.train.Saver(variables)
        with sess.as_default():
            saver.restore(sess,self.config.pretrained_facenet_model)
        return total_loss, variables, logit_variables



