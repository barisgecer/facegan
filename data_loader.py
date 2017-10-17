import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np
import pickle
import os.path

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):

    if os.path.isfile(root +"/list.txt"):
        with open(root +"/list.txt", "rb") as fp:
            paths = pickle.load(fp)
    else:
        for ext in ["jpg", "png"]:
            paths = glob("{}/*/*.{}".format(root, ext))
            if len(paths) != 0:
                with open(root +"/list.txt", "wb") as fp:
                    pickle.dump(paths, fp)
                break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf.image.decode_image(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='real_inputs')

    #queue = tf.image.crop_to_bounding_box(queue, 100, 50, 78, 78)
    queue = tf.image.resize_bilinear(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)


def get_syn_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):

    labels = []
    if os.path.isfile(root +"/list.txt"):
        with open(root +"/list.txt", "rb") as fp:
            paths = pickle.load(fp)
        with open(root +"/labels.txt", "rb") as fp:
            labels = pickle.load(fp)
        with open(root +"/latentvars.txt", "rb") as fp:
            latentvars = pickle.load(fp)
    else:
            paths = glob("{}/*/*.{}".format(root, 'txt'))
            if len(paths) != 0:
                with open(root +"/list.txt", "wb") as fp:
                    pickle.dump([p.replace('txt', 'jpg') for p in paths], fp)

                for im in paths:
                    labels.append(int(im.replace('\\', '/').split('/')[-2]))
                with open(root +"/labels.txt", "wb") as fp:
                    pickle.dump(labels, fp)

                latentvars = np.zeros((len(paths), 451), dtype=np.float32)
                for i, latentvar in enumerate([p for p in paths]):
                    with open(latentvar) as file:
                        latentvars[i, :] = str.split(file.read(), "\n")[0:-1]
                with open(root +"/latentvars.txt", "wb") as fp:
                    pickle.dump(latentvars, fp)

    n_id = max(labels)

    #with Image.open(paths[0]) as img:
    #    w, h = img.size
    #    shape = [h, w, 3]

    #images = tf.convert_to_tensor(list(paths))
    labels = tf.convert_to_tensor(labels)
    latentvars = tf.convert_to_tensor(latentvars)

    # Makes an input queue
    #input_queue = tf.train.slice_input_producer([images, labels, latentvars], shuffle=False, seed=seed)
    input_queue = tf.train.slice_input_producer([labels, latentvars], shuffle=False, seed=seed)
    #reader = tf.WholeFileReader()
    #filename, data = reader.read(input_queue[0])
    #image = tf.image.decode_image(tf.read_file(input_queue[0]), channels=3)
    label = input_queue[0]#[1]
    #reader = tf.TextLineReader()
    #_, latentvar = reader.read(input_queue[2])
    #latentvar = tf.cast(tf.string_split(latentvar,"\n"),tf.float32)
    latentvar = input_queue[1]#[2] #tf.read_file(input_queue[2])

    #filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    #reader = tf.WholeFileReader()
    #filename, data = reader.read(filename_queue)
    #image = tf_decode(data, channels=3)

    #if is_grayscale:
    #    image = tf.image.rgb_to_grayscale(image)
    #image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue_latentvar, queue_label = tf.train.shuffle_batch(
        [latentvar, label], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs',seed=seed)

    #queue_image = tf.image.crop_to_bounding_box(queue_image, 34, 34, 64, 64)
    #queue_image = tf.image.resize_bilinear(queue_image, [scale_size, scale_size])

    #if data_format == 'NCHW':
    #    queue_image = tf.transpose(queue_image, [0, 3, 1, 2])
    #elif data_format == 'NHWC':
    #    pass
    #else:
    #    raise Exception("[!] Unkown data_format: {}".format(data_format))

    return n_id, queue_label,queue_latentvar, n_id


def get_3dmm_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):

    if os.path.isfile(root +"/list.txt"):
        with open(root +"/list.txt", "rb") as fp:
            paths = pickle.load(fp)
        with open(root +"/latentvars.txt", "rb") as fp:
            latentvars = pickle.load(fp)
    else:
        for ext in ["jpg", "png"]:
            paths = glob("{}/*.{}".format(root, ext))
            if len(paths) != 0:
                with open(root +"/list.txt", "wb") as fp:
                    pickle.dump(paths, fp)
                break

        latentvars = np.zeros((len(paths), 512), dtype=np.float32)
        for i, latentvar in enumerate([p.replace(root, root+'/3dmm').replace('.jpg','_noise.txt') for p in paths]):
            with open(latentvar) as file:
                latentvars[i, :] = str.split(file.read(), "\n")[0:-1]
        with open(root + "/latentvars.txt", "wb") as fp:
            pickle.dump(latentvars, fp)


    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]


    images = tf.convert_to_tensor(list(paths))

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, latentvars], shuffle=False, seed=seed)
    #reader = tf.WholeFileReader()
    #filename, data = reader.read(input_queue[0])
    image = tf.image.decode_image(tf.read_file(input_queue[0]), channels=3)
    latentvar = input_queue[1]
    #label = input_queue[1]
    #reader = tf.TextLineReader()
    #_, latentvar = reader.read(input_queue[2])
    #latentvar = tf.cast(tf.string_split(latentvar,"\n"),tf.float32)

    #filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    #reader = tf.WholeFileReader()
    #filename, data = reader.read(filename_queue)
    #image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue_image, latentvar = tf.train.shuffle_batch(
        [image, latentvar ], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='real_3dmm_inputs')

    #queue_image = tf.image.crop_to_bounding_box(queue_image, 34, 34, 64, 64)
    queue_image = tf.image.resize_bilinear(queue_image, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue_image = tf.transpose(queue_image, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue_image), latentvar