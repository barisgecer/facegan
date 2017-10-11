from __future__ import print_function

import os
from io import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from modules import ModuleC
from utils import save_image
from cycleGen import Generator
from operator import itemgetter
import cv2

#denemes
def next(loader):
    return loader.next()[0].data.numpy()


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    image = image / 127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def denorm_img(norm, data_format ='NHWC'):
    return tf.clip_by_value(to_nhwc((norm + 1) * 127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


class Trainer(object):
    def __init__(self, config, data_loader, syn_image, syn_label, syn_latent,image_3dmm, annot_3dmm):
        self.config = config
        self.data_loader = data_loader
        self.syn_image = syn_image
        self.syn_label = syn_label
        self.syn_latent = syn_latent
        self.image_3dmm = image_3dmm
        self.annot_3dmm = annot_3dmm
        self.dataset = config.dataset
        self.n_id_exam_id = config.num_log_id
        self.n_im_per_id = config.num_log_samples

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
                                     name='d_lr_update')

        self.n_id = config.n_id

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
            get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        pretrained_var = self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        #tf.initialize_all_variables()
        pre_train_saver = tf.train.Saver(pretrained_var)
        def load_pretrain(sess):
            pre_train_saver.restore(sess, self.config.pretrained_facenet_model)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=300,
                                 global_step=self.step,
                                 ready_for_local_init_op=None,
                                 init_fn=load_pretrain)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train_renderer(self):
        for step in trange(self.start_step, int(self.max_step/10)):
            fetch_dict = {
                "ren_reg_optim": self.ren_reg_optim,
                "summary": self.summary_op,
            }
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

    def train(self):
        #z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        #alpha_id_fixed = np.repeat(np.random.randint(self.n_id, size=(int(np.floor(self.batch_size / 4.0)),1)) + 1, 4,0)

        #x_fixed = self.get_image_from_loader(self.data_loader)
        #save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        syn_fixed, syn_fixed_label = self.get_fixed_images(self.n_id_exam_id, self.n_im_per_id)
        save_image(syn_fixed, '{}/syn_fixed.png'.format(self.model_dir),nrow=self.n_im_per_id)

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                #print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                #      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            if step % (self.log_step * self.save_step) == 0:
                x_fake = self.generate(syn_fixed, syn_fixed_label, self.model_dir, idx=step)
                #self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                # cur_measure = np.mean(measure_history)
                # if cur_measure > prev_measure * 0.99:
                # prev_measure = cur_measure

    # TODO: Refiner Netork
    def build_model(self):
        def G(input):
            reuse = False
            if hasattr(self, 'G_var'):
                reuse = True
            output, self.G_var = GeneratorCNN(input, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=reuse)
            return output

        def G_inv(input):
            reuse = False
            if hasattr(self, 'G_inv_var'):
                reuse = True
            output, self.G_inv_var = RegressionCNN(input, self.conv_hidden_num, self.syn_latent.get_shape()[1].value + self.z_num ,
                self.repeat_num, self.data_format, reuse=reuse)
            return output

        def R(input):
            reuse = False
            if hasattr(self, 'R_var'):
                reuse = True
            output, self.R_var = Generator('R_inf', True, ngf=32, norm='instance', image_size=self.input_scale_size,reuse=reuse)(input)
            #AddRealismLayers(input,self.conv_hidden_num,4,self.data_format,reuse=reuse)
            return output

        def R_inv(input, isTraining= True):
            reuse = False
            if hasattr(self, 'R_inv_var'):
                reuse = True
            output, self.R_inv_var = Generator('R_inv', isTraining, ngf=32, norm='instance', image_size=self.input_scale_size, reuse=reuse)(input)
            # AddRealismLayers(input,self.conv_hidden_num,4,self.data_format,reuse=reuse,inv=True)
            return output

        # Define Variables
        self.u = self.data_loader # unlabeled real examples
        u_norm = norm_img(self.u)
        c = self.syn_label # identity label
        #self.z = tf.random_uniform((tf.shape(self.syn_latent)[0], self.z_num), minval=-1.0, maxval=1.0) #noise vector
        #p = tf.concat([self.syn_latent,self.z],1)# 3DMM parameters
        #c_onehot = tf.squeeze(tf.one_hot(c, depth=self.n_id, on_value=1.0, off_value=0.0),1)

        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        self.s = self.syn_image
        mask = tf.cast(tf.greater(self.s, 0), tf.float32)
        s_norm = norm_img(self.s)


        # Build Graph
        #y = G(p)
        x = R(s_norm) #R(y)
        y_ = R_inv(x, False)
        #p_ = G_inv(y_)
        self.x = denorm_img(x)

        # TODO: Patch-basaed Discriminator
        # TODO: History of generated images
        d_out, self.D_z, self.D_var = DiscriminatorCNN(
            tf.concat([x, u_norm], 0), self.channel, self.z_num, self.repeat_num,
            self.conv_hidden_num, self.data_format)
        AE_x, AE_u = tf.split(d_out, 2)
        self.AE_x, self.AE_u = denorm_img(AE_x), denorm_img(AE_u)

        C_input = tf.image.resize_bilinear(x, [160, 160])
        C_input = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), C_input)
        C = ModuleC(self.config)
        self.c_loss, self.C_var, self.C_logits_var = \
            C.getNetwork(image=C_input,label_batch=self.syn_label,nrof_classes=self.n_id)


        # Loss functions
        forward_cycle_loss = tf.reduce_mean(tf.abs( s_norm - y_))#p - p_ ))
        backward_cycle_loss = tf.reduce_mean(tf.abs( x - R(y_))) #R(G(p_)) ))
        cycle_loss = forward_cycle_loss + backward_cycle_loss
        #render_loss = tf.reduce_mean(mask * (tf.abs(y - s_norm) + tf.abs(y_ - s_norm)))
        #ren_reg_loss = tf.reduce_mean(mask * (tf.abs(y - s_norm))) + tf.reduce_mean(tf.abs(p - G_inv(s_norm)))
            # Adversarial Training
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_u - u_norm))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_x - x))

        g_reg_loss = tf.reduce_mean(mask * (tf.abs(x - s_norm)))

        # Optimization
        optimizer = tf.train.AdamOptimizer
        g_optimizer, ren_reg_optimizer, d_optimizer = optimizer(self.g_lr),optimizer(self.g_lr), optimizer(self.d_lr)

        #self.ren_reg_optim = ren_reg_optimizer.minimize(ren_reg_loss, global_step=self.step,
                                                        #var_list=self.G_var + self.G_inv_var )

        self.g_optim = g_optimizer.minimize(self.g_loss +  0.5* g_reg_loss + 0.005*self.c_loss+ self.config.lambda_cycle *cycle_loss # + self.config.lambda_ren *render_loss
                                            , global_step=self.step,
                                            var_list=self.R_var + self.R_inv_var + self.C_logits_var )

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, self.g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        kernel = self.C_logits_var[0][0,0]  #
        #x_min = tf.reduce_min(kernel)
        #x_max = tf.reduce_max(kernel)
        #kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        #kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        self.summary_op = tf.summary.merge([
            tf.summary.image("Real Images", self.u),
            tf.summary.image("Rendered Images", self.s),
            tf.summary.image("3dmm Images", self.image_3dmm),
            tf.summary.image("3dmm Annot", self.annot_3dmm),
            #tf.summary.image("Y", y),
            tf.summary.image("Y_", denorm_img(y_)),
            tf.summary.image("Generated Images", self.x),
            #tf.summary.image("filters", kernel_transposed),
            #tf.summary.image("F", tf.slice(F_conv,[0,0,0,0],[8,61,61,1])),
            tf.summary.image("AE_x", self.AE_x),
            tf.summary.image("AE_u", self.AE_u),

            tf.summary.scalar("misc/filter", kernel),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/c_loss", self.c_loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/cycle_loss", cycle_loss),
            tf.summary.scalar("loss/g_reg_loss", g_reg_loss),
            #tf.summary.scalar("loss/ren_reg_loss", ren_reg_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

        return self.C_var

    def build_test_model(self):
        a=2
        # with tf.variable_scope("test") as vs:
        #     # Extra ops for interpolation
        #     #z_optimizer = tf.train.AdamOptimizer(0.0001)
        #
        #     #self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
        #     #self.z_r_update = tf.assign(self.z_r, self.z)
        #
        # G_z_r, _ = GeneratorCNN(
        #     self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)
        #
        # with tf.variable_scope("test") as vs:
        #     self.z_r_loss = tf.reduce_mean(tf.abs(self.u - G_z_r))
        #     self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])
        #
        # test_variables = tf.contrib.framework.get_variables(vs)
        # self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, alpha_id_fix, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.x, {self.s: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path,nrow=self.n_im_per_id)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            # if img.shape[3] in [1, 3]:
            #    img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_u, {self.u: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.u: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_u, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size / 2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.u: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        syn_fixed, syn_fixed_label = self.get_fixed_images(self.n_id_exam_id, self.n_im_per_id)
        self.generate(syn_fixed, syn_fixed_label, self.model_dir, idx='test')

        # root_path = "./"  # self.model_dir
        #
        # all_G_z = None
        # for step in range(3):
        #     real1_batch = self.get_image_from_loader(self.data_loader)
        #     real2_batch = self.get_image_from_loader(self.data_loader)
        #
        #     save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
        #     save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))
        #
        #     self.autoencode(
        #         real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
        #     self.autoencode(
        #         real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))
        #
        #     self.interpolate_G(real1_batch, step, root_path)
        #     # self.interpolate_D(real1_batch, real2_batch, step, root_path)
        #
        #     z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        #     G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))
        #
        #     if all_G_z is None:
        #         all_G_z = G_z
        #     else:
        #         all_G_z = np.concatenate([all_G_z, G_z])
        #     save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))
        #
        # save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_fixed_images( self, nId , nImage):
        return np.array([cv2.imread(self.config.syn_data_dir + "/{:05d}/{:05d}.jpg".format(id + 1, im + 1))[..., ::-1]\
          for id in np.arange(nId) for im in np.arange(nImage)]), [id+1 for id in np.arange(nId) for im in np.arange(nImage)]


    def get_image_from_loader(self, image_loader, label_loader, nId , nImage):
        result = self.sess.run({'img': image_loader, 'label': label_loader})
        x = result['img']
        y = result['label']
        np.random.seed(0)
        while True:
            result = self.sess.run({'img':image_loader,'label':label_loader})
            x = np.append(x,result['img'],axis=0)
            y = np.append(y,result['label'])
            unique, counts = np.unique(y,return_counts=True)
            #if all(np.in1d(np.arange(nId)+1, unique, assume_unique=True)):
            #    if all(counts[0:nId] > nImage-1):
            #        break
            idx = np.sort(counts)[::-1]
            if len(idx)> nId-1:
                if idx[nId-1] > nImage-1:
                    break

        sampleInd = []
        for c in unique[np.argsort(counts)[::-1][:nId]]:
            sampleInd.extend([i for i, z in enumerate(y.tolist()) if z == c][:nImage])

        x_ = x[sampleInd]
        if self.data_format == 'NCHW':
            x_ = x_.transpose([0, 2, 3, 1])
        return x_, y[sampleInd]
