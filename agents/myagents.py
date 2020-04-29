from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from agents.simple_agent import RandomAgent, ForwardOnlyAgent
from agents.rl_agent import SACAgent

import matplotlib.pyplot as plt
import os
import time

from train import get_brain
from common_net import load_from_file, count_number_trainable_params
from preprocess import sim2mapping
from visualize_mapping import plot_viewpoints, mapping_visualizer

import numpy as np
import tensorflow as tf

try:
    import ipdb as pdb
except Exception:
    import pdb

class ExpertAgent(RandomAgent):
    def act(self, observations):
        # print (observations.keys())
        # return super(MyAgent, self).act(observations)
        # plt.figure(1)
        # trav_map = observations["trav_map"]
        # plt.imshow(trav_map)
        # plt.figure(2)
        # plt.imshow(observations["rgb"])
        # plt.show()
        # plt.ginput(timeout=0.01)

        return observations["expert_action"]


class MappingAgent(RandomAgent):
    def __init__(self, params):
        super(MappingAgent, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.params = params

        params.batchdata = 1
        params.trajlen = 1
        sensor_ch = (1 if params.mode == 'depth' else (3 if params.mode == 'rgb' else 4))
        self.global_map_size = (600, 600)

        # Build brain
        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"), tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # dataflow input
                train_brain = get_brain(params.brain, params)
                req = train_brain.requirements()

                # global_map = tf.zeros((1, ) + self.global_map_size + (1, ), dtype=tf.float32)
                self.true_map_input = tf.placeholder(shape=(600, 600, 1), dtype=tf.uint8)
                self.images_input = tf.placeholder(shape=req.sensor_shape + (sensor_ch,), dtype=tf.float32)
                self.xy_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                self.yaw_input = tf.placeholder(shape=(1, ), dtype=tf.float32)
                # self.action_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                actions = tf.zeros((1, 1, 2), dtype=tf.float32)
                self.global_map_input = tf.placeholder(shape=self.global_map_size + (2, ), dtype=tf.float32)

                self.inference_outputs = train_brain.sequential_inference(
                    self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                    actions, prev_global_map_logodds=self.global_map_input[None],
                    is_training=True)
                # global_map, images, xy, yaw, actions = train_data[:5]
                # train_outputs = train_brain.sequential_inference(
                #     global_map, images, xy, yaw, actions, is_training=True)

                # Add the variable initializer Op.
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            count_number_trainable_params(verbose=True)

            # training session
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            self.sess.run(init)

            load_from_file(self.sess, params.load, partialload=params.partialload, loadcenter=[],
                           skip=params.loadskip, autorename=False)


        self.global_map_logodds = None
        self.step_i = 0
        self.reset()


    def __del__(self):
        try:
            self.sess.close()
        except:
            pass

    def reset(self):
        self.global_map_logodds = np.zeros(self.global_map_size + (2,), np.float32)
        self.step_i = 0

    @staticmethod
    def inverse_logodds(x):
        return 1. - 1. / (1 + np.exp(x))

    def act(self, observations):
        # print (observations.keys())
        # return super(MyAgent, self).act(observations)
        # plt.figure(1)
        # trav_map = observations["trav_map"]
        # plt.imshow(trav_map)
        # plt.figure(2)
        # plt.imshow(observations["rgb"])
        # plt.show()
        # plt.ginput(timeout=0.01)

        assert self.params.mode == "both"

        global_map_label = observations["trav_map"]
        # TODO mapping label comes from floor_scan_0.png, this might have extra dispersion

        xy = observations["pose"][:2]
        yaw = observations["rpy"][2:3]
        depth = observations["depth"]
        rgb = observations["rgb"]
        images = np.concatenate([depth, rgb], axis=-1)  # these are 0..1  float format

        images = (images * 255).astype(np.uint8)
        images = np.array(images, np.float32)
        # images = images * 255  # to unit8 0..255 format
        images = images * (2. / 255.) - 1.  # to network input -1..1 format

        xy = sim2mapping(xy=xy)
        yaw = sim2mapping(yaw=np.array(yaw))
        global_map_label = global_map_label[:, :, :1]  # input it as uint8
        assert global_map_label.dtype == np.uint8
        global_map_label = sim2mapping(global_map=global_map_label)
        true_map_input = np.zeros((600, 600, 1), np.uint8)
        global_map_label = global_map_label[:600, :600]
        true_map_input[:global_map_label.shape[0], :global_map_label.shape[1]] = global_map_label

        feed_dict = {
            self.images_input: images, self.xy_input: xy, self.yaw_input: yaw,
            self.global_map_input: self.global_map_logodds,
            self.true_map_input: true_map_input,
        }

        outputs = self.sess.run(self.inference_outputs, feed_dict=feed_dict)

        self.global_map_logodds = np.array(outputs.global_map_logodds[0, -1]) # squeeze
        # global_map_true = self.inverse_logodds(self.global_map_logodds[:, :, 0])
        global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1])
        local_map_label = outputs.local_map_label[0, 0, :, :, 0]
        local_map_pred = outputs.local_map_pred[0, 0, :, :, 0]

        if self.step_i % 10 == 0:
            # Visualize agent
            # assert global_map_label.shape[-1] == 3
            global_map_label = np.concatenate([global_map_label, np.zeros_like(global_map_label), np.zeros_like(global_map_label)], axis=-1)

            plt.figure("Global map label")
            plt.imshow(global_map_label)
            plot_viewpoints(xy[0], xy[1], yaw)
            plt.savefig('./temp-global-map-label.png')

            plt.figure("Global map (%d)"%self.step_i)
            global_map_pred = np.stack([global_map_pred, np.zeros_like(global_map_pred), np.zeros_like(global_map_pred)], axis=-1)
            plt.imshow(global_map_pred)
            plot_viewpoints(xy[0], xy[1], yaw)
            plt.savefig('./temp-global-map-pred.png')

            depth, rgb = mapping_visualizer.recover_depth_and_rgb(images)

            images_fig, images_axarr = plt.subplots(2, 2, squeeze=True)
            plt.axes(images_axarr[0, 0])
            plt.imshow(depth)
            plt.axes(images_axarr[0, 1])
            plt.imshow(rgb)
            plt.axes(images_axarr[1, 0])
            plt.imshow(local_map_pred)
            plt.axes(images_axarr[1, 1])
            plt.imshow(local_map_label)
            plt.savefig('./temp-inputs.png')

            pdb.set_trace()

            plt.close('all')

        self.step_i += 1

        return observations["expert_action"]


def get_agent(agent_class, ckpt_path=""):
    return ExpertAgent()

    if agent_class == "Random":
        return RandomAgent()
    elif agent_class == "ForwardOnly":
        return ForwardOnlyAgent()
    elif agent_class == "SAC":
        return SACAgent(root_dir=ckpt_path)
