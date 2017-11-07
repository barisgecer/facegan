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
import pickle
import os.path
from PIL import Image
from buffer import Buffer

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
    def __init__(self, config, real_image, syn_image, syn_label, syn_latent, image_3dmm, annot_3dmm, latent_3dmm):#, image_3dmm_test, annot_3dmm_test, latent_3dmm_test):
        self.config = config
        self.real_image = real_image
        self.syn_image = syn_image
        self.syn_label = syn_label
        self.syn_latent = syn_latent
        self.image_3dmm = image_3dmm
        self.annot_3dmm = annot_3dmm
        self.latent_3dmm = latent_3dmm
        #self.image_3dmm_test = image_3dmm_test
        #self.annot_3dmm_test = annot_3dmm_test
        #self.latent_3dmm_test = latent_3dmm_test
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
        self.ren_lr = tf.Variable(config.ren_lr, name='ren_lr')
        self.reg_lr = tf.Variable(config.reg_lr, name='reg_lr')

        self.g_lr_warmup = tf.assign(self.g_lr, tf.minimum(self.g_lr * (1+ (config.num_gpu-1) * (config.log_step/config.warm_up)), config.g_lr * config.num_gpu),
                                     name='g_lr_update')
        self.d_lr_warmup = tf.assign(self.d_lr, tf.minimum(self.d_lr * (1+ (config.num_gpu-1) * (config.log_step/config.warm_up)), config.d_lr * config.num_gpu),
                                     name='d_lr_update')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
                                     name='d_lr_update')
        self.ren_lr_update = tf.assign(self.ren_lr, tf.maximum(self.ren_lr * 0.5, config.lr_lower_boundary),
                                     name='ren_lr_update')
        self.reg_lr_update = tf.assign(self.reg_lr, tf.maximum(self.reg_lr * 0.5, config.lr_lower_boundary),
                                     name='reg_lr_update')
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
            get_conv_shape(self.real_image, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = int(config.max_step / config.num_gpu)
        self.save_step = config.save_step
        self.lr_update_step = int(config.lr_update_step / config.num_gpu)

        self.is_train = config.is_train
        self.gen_var = self.build_model()

        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        self.load_pretrain = None
        if not config.train_generator:
            pre_train_saver = tf.train.Saver(self.gen_var)

            def load_pretrain(sess):
                if not config.train_generator:
                    pre_train_saver.restore(sess, config.pretrained_gen)

            self.load_pretrain = load_pretrain
        self.rng = np.random.RandomState(config.random_seed)
        self.history_buffer = Buffer(config, self.rng)


    def prepare_session(self, var_saved):
        self.saver = tf.train.Saver(var_saved)
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=300,
                                 global_step=self.step,
                                 ready_for_local_init_op=None,
                                 init_fn=self.load_pretrain)

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
        self.prepare_session(self.ren_var)

        for step in trange(self.start_step, int(self.max_step)):
            fetch_dict = {
                "ren_optim": self.ren_optim,
                "summary": self.summary_op,
            }
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.ren_lr_update])

    def train_regressor(self):
        if self.config.pretrain_generator:
            self.prepare_session(self.reg_var + self.gen_var)
        else:
            self.prepare_session(self.reg_var)

        fixed_image, fixed_label, fixed_latent = self.get_fixed_images(self.n_id_exam_id, self.n_im_per_id)
        save_image(fixed_image, '{}/syn_fixed.png'.format(self.model_dir),nrow=self.n_im_per_id)

        for step in trange(self.start_step, int(self.max_step)):
            fetch_dict = {
                "reg_optim": self.reg_optim,
                "summary": self.summary_op,
            }
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if (step % (self.log_step * self.save_step) == 0) & self.config.pretrain_generator:
                x_fake = self.generate(fixed_latent, fixed_label, self.model_dir, idx=step)
                #self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.reg_lr_update])

    def train(self):
        self.prepare_session(None)
        # z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        # alpha_id_fixed = np.repeat(np.random.randint(self.n_id, size=(int(np.floor(self.batch_size / 4.0)),1)) + 1, 4,0)

        fixed_image, fixed_label, fixed_latent = self.get_fixed_images(self.n_id_exam_id, self.n_im_per_id)
        save_image(fixed_image, '{}/syn_fixed.png'.format(self.model_dir), nrow=self.n_im_per_id)

        prev_measure = 1
        #measure_history = deque([0] * self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "output": self.x_all_norm,
                #"measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict,{self.x_hist: self.history_buffer.sample()})

            self.history_buffer.push(result['output'])
            #measure = result['measure']
            #measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} , k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, k_t))
                if step <= self.config.warm_up:
                    self.sess.run([self.g_lr_warmup, self.d_lr_warmup])

            if step % (self.log_step * self.save_step) == 0:
                x_fake = self.generate(fixed_image, fixed_label, self.model_dir, idx=step)
                # self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                self.lr_update_step = int(self.lr_update_step/2)
                # cur_measure = np.mean(measure_history)
                # if cur_measure > prev_measure * 0.99:
                # prev_measure = cur_measure

    def generate_dataset(self):
        with open(self.config.syn_data_dir +"/list.txt", "rb") as fp:
            paths = pickle.load(fp)
        with open(self.config.syn_data_dir +"/labels.txt", "rb") as fp:
            labels = pickle.load(fp)
        with open(self.config.syn_data_dir + "/latentvars.txt", "rb") as fp:
            latentvars = pickle.load(fp)

        save_dir = os.path.join(self.config.data_dir, self.config.save_syn_dataset)
        os.makedirs(save_dir,exist_ok=True)
        self.prepare_session(self.gen_var)
        for i in range(0,len(paths),self.config.batch_size):
            pa = paths[i:min(i + self.config.batch_size, len(paths))]
            inputs = np.array([cv2.imread(pa[j])[..., ::-1] for j in np.arange(len(pa))])
            x = self.sess.run(self.x, {self.syn_image: inputs})
            for im in range(len(x)):
                os.makedirs(os.path.dirname(pa[im].replace(self.config.syn_data_dir,save_dir)),exist_ok=True)
                Image.fromarray(x[im].astype(np.uint8)).save(pa[im].replace(self.config.syn_data_dir,save_dir))

    def build_model(self):
        def average_gradients(tower_grads):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            return average_grads

        with tf.device('/cpu:0'):
            tower_grads_G = []
            tower_grads_G_inv = []
            tower_grads_D = []
            balances1 = []
            balances2 = []
            balances3 = []
            self.x_all = []
            reuse_vars = False

            optimizer = tf.train.AdamOptimizer
            g_optimizer, g_inv_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.g_lr), optimizer(self.d_lr)

            for i in range(self.config.num_gpu):
                gpu_ind = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
                with tf.device('/gpu:%d' % i):
                    def R(input):
                        reuse = reuse_vars
                        if hasattr(self, 'R_var'):
                            reuse = True
                        output, self.R_var = GeneratorCNN("R",input, self.conv_hidden_num, self.channel,
                            self.repeat_num, self.data_format, reuse=reuse)
                        return output

                    # def G(input):
                    #     reuse = False
                    #     if hasattr(self, 'G_var'):
                    #         reuse = True
                    #     output, self.G_var = GeneratorCNN("G_inf",input, self.conv_hidden_num, self.channel,
                    #         self.repeat_num, self.data_format, reuse=reuse)
                    #     return output
                    #
                    # def G_inv(input):
                    #     reuse = False
                    #     if hasattr(self, 'G_inv_var'):
                    #         reuse = True
                    #     output, self.G_inv_var = RegressionCNN("G_inv",input, self.conv_hidden_num, self.syn_latent.get_shape()[1].value,
                    #         self.repeat_num, self.data_format, reuse=reuse)
                    #     return output

                    def G(input):
                        reuse = reuse_vars
                        if hasattr(self, 'G_var'):
                            reuse = True
                        output, self.G_var = Generator('G_inf', True, ngf=self.config.conv_hidden_num_res, norm='instance', image_size=self.input_scale_size,reuse=reuse, drop_keep=0.9)(input)
                        #AddRealismLayers(input,self.conv_hidden_num,4,self.data_format,reuse=reuse)
                        return output

                    def G_inv(input):
                        reuse = reuse_vars
                        if hasattr(self, 'G_inv_var'):
                            reuse = True
                        output, self.G_inv_var = Generator('G_inv', True, ngf=self.config.conv_hidden_num_res, norm='instance', image_size=self.input_scale_size, reuse=reuse)(input)
                        # AddRealismLayers(input,self.conv_hidden_num,4,self.data_format,reuse=reuse,inv=True)
                        return output

                    # Define Variables
                    self.k_t = tf.Variable(0., trainable=False, name='k_t')
                    self.k_t2 = tf.Variable(0., trainable=False, name='k_t2')
                    self.k_t3 = tf.Variable(0., trainable=False, name='k_t3')
                    #self.k_t4 = tf.Variable(0., trainable=False, name='k_t4')
                    real_image_norm = norm_img(self.real_image[gpu_ind])# unlabeled real examples
                    #z = tf.random_normal((tf.shape(self.syn_latent)[0], self.z_num)) #noise vector
                    #self.p = tf.concat([self.syn_latent, z],1)# 3DMM parameters
                    #c = self.syn_label # identity label
                    #c_onehot = tf.squeeze(tf.one_hot(c, depth=self.n_id, on_value=1.0, off_value=0.0),1)
                    #self.s = self.annot_3dmm
                    #mask = tf.cast(tf.greater(self.s, 0), tf.float32)
                    #s_norm = norm_img(self.s)

                    # Build Graph
                    # Generation
                    syn_image = norm_img(self.syn_image[gpu_ind])
                    y_ = G_inv(real_image_norm)
                    #syn_image_noise = tf.concat(syn_image, tf.random_normal((tf.shape(syn_image)[1], tf.shape(syn_image)[2])),3)
                    #y_noise_ = tf.concat(y_, tf.random_normal((tf.shape(y_)[1], tf.shape(y_)[2])),3)
                    x, x_, paired_x = tf.split(G(tf.concat([syn_image,y_,norm_img(self.annot_3dmm[gpu_ind])],0)),3)
                    y, paired_y = tf.split(G_inv(tf.concat([x,norm_img(self.image_3dmm[gpu_ind])],0)),2)
                    self.x = denorm_img(x)
                    self.x_all.append(x)

                    # Rendering
                    #ren_syn = R(self.syn_latent)
                    #ren_p = R(self.syn_latent)
                    #ren_p_ = R(p_) #tf.split(p_, [451, 61],1)[0]

                    # Regression
                    #ren_reg = G_inv(norm_img(self.image_3dmm))
                    #ren_reg = R(reg_latent)

                    #ren_reg_test = G_inv(norm_img(self.image_3dmm_test))

                    def D(name,x, real_image_norm, k_t, reuse=False, two_x = False):
                        # TO-DO: Patch-based Discriminator
                        # TO-DO: History of generated images
                        d_out, self.D_z, D_var = DiscriminatorCNN(name,
                            tf.concat([x, real_image_norm], 0), self.channel, self.z_num, self.repeat_num,
                            self.conv_hidden_num, self.data_format,(reuse | reuse_vars))
                        if two_x:
                            AE_x1, AE_x2, AE_u = tf.split(d_out, 3)
                            AE_x = tf.concat([AE_x1, AE_x2],0)
                        else:
                            AE_x, AE_u = tf.split(d_out, 2)
                            AE_x1 = AE_x
                        #self.AE_x, self.AE_u = denorm_img(AE_x), denorm_img(AE_u)

                        # Loss functions
                        # Adversarial Training
                        d_loss_real = tf.reduce_mean(tf.abs(AE_u - real_image_norm))
                        g_loss = tf.reduce_mean(tf.abs(AE_x - x))
                        d_loss = d_loss_real - k_t * g_loss
                        balance = self.gamma * d_loss_real - g_loss
                        return d_loss, g_loss, balance, D_var, AE_x1 , AE_u

                    #self.p_loss = tf.reduce_mean(tf.abs(real_image_norm - x_))
                    self.s_loss = tf.reduce_mean(tf.abs(syn_image - y))

                    sd_loss_real_forw = tf.reduce_mean(tf.abs(paired_y - norm_img(self.annot_3dmm[gpu_ind])))
                    sd_loss_forw = sd_loss_real_forw - self.k_t3 * self.s_loss
                    balance3 = self.gamma * sd_loss_real_forw - self.s_loss

                    #sd_loss_real_back = tf.reduce_mean(tf.abs(paired_x - norm_img(self.image_3dmm[gpu_ind])))
                    #sd_loss_back = sd_loss_real_back - self.k_t4 * self.p_loss
                    #balance4 = self.gamma * sd_loss_real_back - self.p_loss

                    # Pretrain
                    #self.ren_loss = tf.reduce_mean(tf.abs(ren_syn - norm_img(self.syn_image)))
                    #self.reg_loss = tf.reduce_mean(tf.abs(ren_reg - norm_img(self.annot_3dmm)))
                    #self.reg_test_loss = tf.reduce_mean(tf.abs(ren_reg_test - norm_img(self.annot_3dmm_test)))
                    #self.reg_latent_loss = tf.reduce_mean(tf.abs(reg_latent - tf.split(self.latent_3dmm, [451, 61],1)[0]))
                    self.x_hist = tf.placeholder(tf.float32, [None, None, None, 3], 'x_hist')
                    d_loss_forw, g_loss_forw, balance, D_var_forw, self.AE_x, self.AE_u = D("D_forw",tf.concat([x, self.x_hist],0), real_image_norm, self.k_t, two_x=True)
                    d_loss_back, g_loss_back, balance2, D_var_back, _, _ = D("D_back",tf.concat([y, y_],0), syn_image, self.k_t2, two_x=True)
                    self.g_loss = g_loss_forw + g_loss_back
                    self.d_loss = d_loss_forw + d_loss_back

                    # Optimization

                    #self.ren_optim = g_optimizer.minimize(self.ren_loss, global_step=self.step,var_list=self.R_var )

                    #self.reg_optim = g_optimizer.minimize(self.reg_loss, global_step=self.step,var_list=self.G_inv_var )

                    g_optim = g_optimizer.compute_gradients( g_loss_forw + self.config.lambda_s *(self.s_loss), var_list=self.G_var )

                    g_inv_optim = g_inv_optimizer.compute_gradients(g_loss_back + self.config.lambda_d*sd_loss_forw + self.config.lambda_s *(self.s_loss), var_list=self.G_inv_var )

                    d_optim = d_optimizer.compute_gradients(self.d_loss, var_list=D_var_forw + D_var_back)

                    tower_grads_G.append(g_optim)
                    tower_grads_G_inv.append(g_inv_optim)
                    tower_grads_D.append(d_optim)
                    balances1.append(balance)
                    balances2.append(balance2)
                    balances3.append(balance3)

                    self.balance = balance #self.gamma * self.d_loss_real - self.g_loss
                    #self.measure = self.d_loss_real + tf.abs(self.balance)
                    reuse_vars = True

                    #kernel = G_conv_var[0]  #
                    #x_min = tf.reduce_min(kernel)
                    #x_max = tf.reduce_max(kernel)
                    #kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
                    #kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
                    if i == self.config.num_gpu-1:
                        self.summary_op = tf.summary.merge([
                            tf.summary.image("Real Images", self.real_image[gpu_ind]),
                            tf.summary.image("Generated Images", self.x),
                            #tf.summary.image("Intended Rendering", denorm_img(ren_p)),
                            tf.summary.image("Generated Rendering", denorm_img(y)),
                            #tf.summary.image("Regressor Input", self.image_3dmm),
                            #tf.summary.image("Regressor Output", denorm_img(ren_reg)),
                            #tf.summary.image("Regressor GT", self.annot_3dmm),
                            #tf.summary.image("Regressor Input-Test", self.image_3dmm_test),
                            #tf.summary.image("Regressor Output-Test", denorm_img(ren_reg_test)),
                            #tf.summary.image("Regressor GT-Test", self.annot_3dmm_test),
                            #tf.summary.image("Rendering Output", denorm_img(ren_syn)),
                            tf.summary.image("Rendering GT", self.syn_image[gpu_ind]),
                            #tf.summary.image("filters", kernel_transposed),
                            tf.summary.image("AE_x", self.AE_x),
                            tf.summary.image("AE_u", self.AE_u),

                            tf.summary.scalar("loss/d_loss", self.d_loss),
                            tf.summary.scalar("loss/s_loss", self.s_loss),
                            #tf.summary.scalar("loss/p_loss", self.p_loss),
                            tf.summary.scalar("loss/g_loss", self.g_loss),
                            tf.summary.scalar("loss/g_loss_back", g_loss_back),
                            #tf.summary.scalar("loss/sd_loss_back", sd_loss_back),
                            #tf.summary.scalar("loss/ren_loss", self.ren_loss),
                            #tf.summary.scalar("loss/reg_loss", self.reg_loss),
                            #tf.summary.scalar("loss/reg_test_loss", self.reg_test_loss),
                            #tf.summary.scalar("loss/reg_latent_loss", self.reg_latent_loss),
                            #tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
                            #tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
                            #tf.summary.scalar("misc/measure", self.measure),
                            tf.summary.scalar("misc/k_t", self.k_t),
                            tf.summary.scalar("misc/k_t2", self.k_t2),
                            tf.summary.scalar("misc/d_lr", self.d_lr),
                            tf.summary.scalar("misc/g_lr", self.g_lr),
                            tf.summary.scalar("misc/balance", tf.reduce_mean(balances1)),
                        ])
                    #tf.get_variable_scope().reuse_variables()

            tower_grads_G = average_gradients(tower_grads_G)
            tower_grads_G_inv = average_gradients(tower_grads_G_inv)
            tower_grads_D = average_gradients(tower_grads_D)
            train_op_G = g_optimizer.apply_gradients(tower_grads_G,self.step)
            train_op_G_inv = g_inv_optimizer.apply_gradients(tower_grads_G_inv)
            train_op_D = d_optimizer.apply_gradients(tower_grads_D)

            #variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.step)
            #variables_averages_op = variable_averages.apply(tf.trainable_variables())
            self.x_all_norm = tf.concat(self.x_all,0)
            self.x_all = denorm_img(self.x_all_norm)

            with tf.control_dependencies([train_op_G, train_op_G_inv, train_op_D]):
                self.k_update = tf.assign(
                    self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * tf.reduce_mean(balances1), 0, 1))
                self.k_update2 = tf.assign(
                    self.k_t2, tf.clip_by_value(self.k_t2 + self.lambda_k * tf.reduce_mean(balances2), 0, 1))
                self.k_update3 = tf.assign(
                    self.k_t3, tf.clip_by_value(self.k_t3 + self.lambda_k * tf.reduce_mean(balances3), 0, 1))
                #self.k_update4 = tf.assign(
                #    self.k_t4, tf.clip_by_value(self.k_t4 + self.lambda_k * (balance4), 0, 1))

        return self.G_var#, self.G_inv_var, self.G_var

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
        with tf.device('/gpu:0'):
            x = np.array([self.sess.run(self.x_all, {self.syn_image: inputs[i:min(i + self.config.batch_size*self.config.num_gpu, len(inputs))]}) for i in range(0,len(inputs),self.config.batch_size*self.config.num_gpu)])
            x = x.reshape((-1,)+x.shape[2:])
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
        def readfile(file_path):
            with open(file_path) as file:
                return np.array([float(i) for i in str.split(file.read(), "\n")[0:-1]])

        images = np.array([cv2.imread(self.config.syn_data_dir + "/{:05d}/{:05d}.jpg".format(id + 1, im + 1))[..., ::-1]\
          for id in np.arange(nId) for im in np.arange(nImage)])
        labels = [id+1 for id in np.arange(nId) for im in np.arange(nImage)]
        latents = [np.concatenate((readfile(self.config.syn_data_dir + "/{:05d}/{:05d}.txt".format(id + 1, im + 1)), np.random.randn(self.z_num)))\
          for id in np.arange(nId) for im in np.arange(nImage)]

        return images, labels, latents


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
