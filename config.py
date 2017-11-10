#-*- coding: utf-8 -*-
import argparse
from pygit2 import Repository
import os.path

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_branch_name():
    if (os.name!='nt') & (os.path.isfile("branchname.txt")):
        with open("branchname.txt", "rb") as file:
            branchname = str(file.readline().decode("utf-8"))
    else:
        branchname = Repository('.').head.shorthand
        with open("branchname.txt", "w") as file:
            file.write(branchname)
    return branchname

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_scale_size', type=int, default=64,#64
                     help='input image will be resized with the given value as width and height')
net_arg.add_argument('--conv_hidden_num', type=int, default=64,
                     choices=[64, 128],help='n in the paper')
net_arg.add_argument('--conv_hidden_num_res', type=int, default=32,
                     choices=[64, 128],help='n in the paper')
net_arg.add_argument('--z_num', type=int, default=64, choices=[64, 128])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='vgg-aligned64')
data_arg.add_argument('--num_gpu', type=int, default=1)
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=32)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters
train_arg = add_argument_group('Training')

train_arg.add_argument('--cont', type=str, default="",choices=['','ren','reg','gen']) # If set, rest should be False
#train_arg.add_argument('--train_renderer', type=str2bool, default=False)
#train_arg.add_argument('--train_regressor', type=str2bool, default=False)
#train_arg.add_argument('--pretrain_generator', type=str2bool, default=True) # True if not pretrained (when train_reg is False)
train_arg.add_argument('--train_generator', type=str2bool, default=True)
train_arg.add_argument('--generate_dataset', type=str2bool, default=True)
train_arg.add_argument('--fit_dataset', type=str2bool, default=False)

#train_arg.add_argument('--pretrained_ren', type=str, default='pretrained_models/ren')
#train_arg.add_argument('--pretrained_reg', type=str, default='pretrained_models/reg')
train_arg.add_argument('--pretrained_gen', type=str, default='pretrained_models/gen')
train_arg.add_argument('--save_syn_dataset', type=str, default='generated')
train_arg.add_argument('--save_fitting', type=str, default='fitting')
train_arg.add_argument('--pretrained_rec', type=str, default='facenet_model/model-20170511-185253.ckpt-80000', help='Pretrained facenet model')

train_arg.add_argument('--task', type=str, default=get_branch_name(), help='default branch name')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=248000)
train_arg.add_argument('--lr_update_step', type=int, default=128000)
train_arg.add_argument('--warm_up', type=int, default=5000)
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--ren_lr', type=float, default=0.00008)
train_arg.add_argument('--reg_lr', type=float, default=0.00008)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--lambda_d', type=float, default=0.1, help='')
train_arg.add_argument('--lambda_s', type=float, default=0.2, help='')
train_arg.add_argument('--lambda_c', type=float, default=0.0, help='')
train_arg.add_argument('--lambda_a', type=float, default=0.02, help='')

train_arg.add_argument('--method_c', type=str, default="none",choices=['none','magnet','softmax','center'])
# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=20)
misc_arg.add_argument('--num_log_id', type=int, default=16)
misc_arg.add_argument('--num_log_samples', type=int, default=8)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='c:/data')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)

# FaceGAN
misc_arg.add_argument('--n_id', type=int, default=10, help='Number of different identities to be generated')
misc_arg.add_argument('--facenet_scope', type=str, default='InceptionResnetV1', help='Facenet model scope name defined in --model_def')
parser.add_argument('--syn_dataset', type=str,
                    help='Directory where synthetic images generated by 3dmm', default='syn_data-aligned')
parser.add_argument('--dataset_3dmm_test', type=str,
                    help='Directory where synthetic images generated by 3dmm', default='AFLW-test')
net_arg.add_argument('--syn_scale_size', type=int, default=64,
                     help='input image will be resized with the given value as width and height')
parser.add_argument('--dataset_3dmm', type=str,
                    help='A real dataset annotated with 3dmm renderings', default='300W-3D-aligned')



# FaceNet
parser.add_argument('--logs_base_dir', type=str,
                    help='Directory where to write event logs.', default='~/logs/facenet')
parser.add_argument('--models_base_dir', type=str,
                    help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
parser.add_argument('--gpu_memory_fraction', type=float,
                    help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
parser.add_argument('--pretrained_model', type=str,
                    help='Load a pretrained model before training starts.')
parser.add_argument('--model_def', type=str,
                    help='Model definition. Points to a module containing the definition of the inference graph.',
                    default='models.inception_resnet_v1')
parser.add_argument('--max_nrof_epochs', type=int,
                    help='Number of epochs to run.', default=500)
parser.add_argument('--image_size', type=int,
                    help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--epoch_size', type=int,
                    help='Number of batches per epoch.', default=1000)
parser.add_argument('--embedding_size', type=int,
                    help='Dimensionality of the embedding.', default=128)
parser.add_argument('--random_crop',
                    help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                         'If the size of the images in the data directory is equal to image_size no cropping is performed',
                    action='store_true')
parser.add_argument('--random_flip',
                    help='Performs random horizontal flipping of training images.', action='store_true')
parser.add_argument('--random_rotate',
                    help='Performs random rotations of training images.', action='store_true')
parser.add_argument('--keep_probability', type=float,
                    help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
parser.add_argument('--weight_decay', type=float,
                    help='L2 weight regularization.', default=5e-5)
parser.add_argument('--center_loss_factor', type=float,
                    help='Center loss factor.', default=0.0)
parser.add_argument('--center_loss_alfa', type=float,
                    help='Center update rate for center loss.', default=0.95)
parser.add_argument('--learning_rate', type=float,
                    help='Initial learning rate. If set to a negative value a learning rate ' +
                         'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
parser.add_argument('--learning_rate_decay_epochs', type=int,
                    help='Number of epochs between learning rate decay.', default=100)
parser.add_argument('--learning_rate_decay_factor', type=float,
                    help='Learning rate decay factor.', default=1.0)
parser.add_argument('--moving_average_decay', type=float,
                    help='Exponential decay for tracking of training parameters.', default=0.9999)
parser.add_argument('--seed', type=int,
                    help='Random seed.', default=666)
parser.add_argument('--nrof_preprocess_threads', type=int,
                    help='Number of preprocessing (data loading and augmentation) threads.', default=4)
parser.add_argument('--log_histograms',
                    help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
parser.add_argument('--learning_rate_schedule_file', type=str,
                    help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                    default='data/learning_rate_schedule.txt')
parser.add_argument('--filter_filename', type=str,
                    help='File containing image data used for dataset filtering', default='')
parser.add_argument('--filter_percentile', type=float,
                    help='Keep only the percentile images closed to its class center', default=100.0)
parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                    help='Keep only the classes with this number of examples or more', default=0)

# Parameters for validation on LFW
parser.add_argument('--lfw_pairs', type=str,
                    help='The file containing the pairs to use for validation.', default='data/pairs.txt')
parser.add_argument('--lfw_file_ext', type=str,
                    help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
parser.add_argument('--lfw_dir', type=str,
                    help='Path to the data directory containing aligned face patches.', default='')
parser.add_argument('--lfw_batch_size', type=int,
                    help='Number of images to process in a batch in the LFW test set.', default=100)
parser.add_argument('--lfw_nrof_folds', type=int,
                    help='Number of folds to use for cross validation. Mainly used for testing.', default=10)



def get_config():
    config, unparsed = parser.parse_known_args()
    setattr(config, 'data_format', 'NHWC')
    return config, unparsed
