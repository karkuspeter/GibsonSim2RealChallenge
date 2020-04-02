from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys, cv2
import tensorflow as tf
import numpy as np

# Fix Python 2.x.
try: input = raw_input
except NameError: pass

try:
    import ipdb as pdb
except Exception:
    import pdb

plt.ion()


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_bytelist_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def py_proc_image(img, resize=None, display=False):
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def raw_images_to_array(images):
    image_list = []
    for image_str in images:
        image = py_proc_image(image_str) #, (56, 56))
        image = np.atleast_3d(image.astype(np.float32))
        image = image / 255.
        image_list.append(image)

    return np.stack(image_list, axis=0)


def display_data(file):
    gen = tf.python_io.tf_record_iterator(file)
    for data_i, string_record in enumerate(gen):
        result = tf.train.Example.FromString(string_record)
        features = result.features.feature

        # poses: (x, y, z) in meters
        poses = features['poses'].bytes_list.value[0]
        poses = np.frombuffer(poses, np.float32).reshape((-1, 3))
        print ("True poses (first three)")
        print (poses[:3])

        # orientations: quaternion
        rpy = features['rpy'].bytes_list.value[0]
        rpy = np.frombuffer(rpy, np.float32).reshape((-1, 3))
        print ("Roll-pitch-yaw (first three)")
        print (rpy[:3])

        # observations are enceded as a list of png images
        rgb = raw_images_to_array(list(features['rgbs'].bytes_list.value))
        depth = raw_images_to_array(list(features['depths'].bytes_list.value))
        if depth.ndim > 3:
            depth = depth.squeeze(axis=-1)

        for i in range(poses.shape[0]):
            print ("x: %f; y:%f; yaw:%f"%(poses[i,0], poses[i,1], rpy[i, 2]))
            plt.figure(1)
            plt.imshow(rgb[i])
            plt.figure(2)
            plt.imshow(depth[i])

            plt.show()

            if input("proceed?") == 'n':
                pdb.set_trace()
                break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage: display_data.py xxx.tfrecords")
        exit()

    display_data(sys.argv[1])