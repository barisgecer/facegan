from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim
import numpy as np


class ModuleC(object):
    def __init__(self, config):
        self.config = config
        self.network = importlib.import_module('facenet.src.'+config.model_def)


    def getNetwork(self,image, nrof_classes, label_batch, reuse= False, is_train= True):
        # Build the inference graph
        prelogits, _ = self.network.inference(image, self.config.keep_probability, phase_train = False, bottleneck_layer_size=self.config.embedding_size, weight_decay=self.config.weight_decay, reuse=reuse)
        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                      scope='Logits', reuse=reuse)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        logit_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Logits')
        with tf.variable_scope("centroids", reuse=reuse):  # reuse the second time
            centroids = tf.get_variable('centers', [nrof_classes, self.config.embedding_size], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0), trainable=False)

        total_loss =  0
        c_loss_each = 0
        # Calculate the total losses
        if self.config.method_c == 'softmax':
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n([cross_entropy_mean], name='total_loss')
        elif self.config.method_c == 'magnet':
            total_loss,c_loss_each,centroids = magnet_loss(embeddings,label_batch,nrof_classes,centroids,center_alpha=self.config.center_loss_alfa,is_train=is_train)
        elif self.config.method_c == 'center':
            total_loss,_ = center_loss(embeddings, label_batch,centroids, self.config.center_loss_alfa, nrof_classes)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.facenet_scope)

        return total_loss, variables, logit_variables, centroids, c_loss_each, embeddings



def magnet_loss(features, classes, nrof_classes, centroids, alpha=1.0, center_alpha=0.95, is_train= True):
    """Compute magnet loss.

    Given a tensor of features `r`, the assigned class for each example,
    the assigned cluster for each example, the assigned class for each
    cluster, the total number of clusters, and separation hyperparameter,
    compute the magnet loss according to equation (4) in
    http://arxiv.org/pdf/1511.05939v2.pdf.

    Note that cluster and class indexes should be sequential startined at 0.

    Args:
        features: A batch of features.
        classes: Class labels for each example.
        clusters: Cluster labels for each example.
        n_clusters: Total number of clusters.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """

    classes = tf.reshape(classes, [-1])
    if not is_train:
        center_alpha = 1.0
    centers_batch = tf.gather(centroids, classes)
    diff = (1 - center_alpha) * (centers_batch - features)

    # Helper to compute boolean mask for distance comparisons
    def comparison_mask(a_labels, b_labels):
        return tf.equal(tf.expand_dims(a_labels, 1),
                        tf.expand_dims(b_labels, 0))

    # Compute squared distance of each example to each cluster centroid
    sample_costs = tf.squared_difference(centroids, tf.expand_dims(features, 1))
    sample_costs = tf.reduce_sum(sample_costs, 2)

    # Select distances of examples to their own centroid
    intra_cluster_mask = comparison_mask(classes, np.arange(nrof_classes, dtype=np.int32))
    intra_cluster_costs = tf.reduce_sum(tf.to_float(intra_cluster_mask) * sample_costs, 1)

    # Compute variance of intra-cluster distances
    N = tf.shape(features)[0]
    variance = tf.reduce_sum(intra_cluster_costs) / tf.to_float(N - 1)
    var_normalizer = -1 / (2 * variance ** 2)

    # Compute numerator
    numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)

    # Compute denominator
    diff_class_mask = tf.to_float(tf.logical_not(intra_cluster_mask))
    denom_sample_costs = tf.exp(var_normalizer * sample_costs)
    denominator = tf.reduce_sum(diff_class_mask * denom_sample_costs, 1)

    # Compute example losses and total loss
    epsilon = 1e-8
    losses = tf.nn.relu(-tf.log(numerator / (denominator + epsilon) + epsilon))
    total_loss = tf.reduce_mean(losses)

    centroids = tf.scatter_sub(centroids, classes, diff)

    return total_loss, losses, centroids


def center_loss(features, label,centers, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers