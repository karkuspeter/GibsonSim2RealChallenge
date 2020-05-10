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
from preprocess import sim2mapping, mapping2sim, object_map_from_states
from visualize_mapping import plot_viewpoints, plot_target_and_path, mapping_visualizer

import numpy as np
import tensorflow as tf

from gibsonagents.expert import Expert
from gibsonagents.classic_mapping import ClassicMapping

# from gibson2.core.physics.scene import BuildingScene
from gibson2.envs.locomotor_env import NavigateEnv
from preprocess import GibsonData
from gibson2.utils.utils import rotate_vector_3d

try:
    import ipdb as pdb
except Exception:
    import pdb


POSE_ESTIMATION_SOURCE = 'velocity'  #''true' # 'velocity'
MAP_SOURCE = 'pred'   # 'label'  # 'pred'
USE_OBJECT_MAP_FOR_TRACK = [False, True, False]
IGNORE_OBJECT_MAP = True
PLOT_EVERY_N_STEP = -1  # 10
START_WITH_SPIN = True



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

        self.sim2real_track = str(os.environ["SIM2REAL_TRACK"])
        print (" ******* \n \n \n *******")
        print ("Initializing agent. %s track."%self.sim2real_track)
        try:
            config_file = os.environ['CONFIG_FILE']
            print (config_file)
            from gibson2.utils.utils import parse_config
            config = parse_config(config_file)
            print (config)

        except:
            print ("Cannot print config.")
        print(" ******* \n \n \n *******")

        # Replace load
        track_i = (0 if self.sim2real_track == 'static' else (1 if self.sim2real_track == 'interactive' else 2))
        load_for_track = params.gibson_load_for_tracks[track_i]
        if load_for_track != '':
            params.load = [load_for_track, ]

        params.object_map = (1 if USE_OBJECT_MAP_FOR_TRACK[track_i] else 0)

        print (params)

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.params = params
        self.pose_estimation_source = POSE_ESTIMATION_SOURCE
        self.map_source = MAP_SOURCE
        self.start_with_spin = START_WITH_SPIN
        self.max_confidence = 0.96   # 0.98
        self.confidence_threshold = None  # (0.2, 0.01)  # (0.35, 0.05)
        self.use_custom_visibility = (self.params.visibility_mask == 2)
        self.use_object_map = (self.params.object_map > 0)  # if want to change also need to overwrite params.object_map

        self.map_ch = (3 if self.use_object_map else 2)
        self.accumulated_spin = 0.
        self.spin_direction = None

        params.batchdata = 1
        params.trajlen = 1
        sensor_ch = (1 if params.mode == 'depth' else (3 if params.mode == 'rgb' else 4))
        self.max_map_size = (800, 800)

        # Build brain
        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"), tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # dataflow input
                train_brain = get_brain(params.brain, params)
                req = train_brain.requirements()
                self.brain_requirements = req
                self.local_map_shape = req.local_map_size

                # global_map = tf.zeros((1, ) + self.global_map_size + (1, ), dtype=tf.float32)
                self.true_map_input = tf.placeholder(shape=self.max_map_size + (1, ), dtype=tf.uint8)
                self.images_input = tf.placeholder(shape=req.sensor_shape + (sensor_ch,), dtype=tf.float32)
                self.xy_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                self.yaw_input = tf.placeholder(shape=(1, ), dtype=tf.float32)
                # self.action_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                actions = tf.zeros((1, 1, 2), dtype=tf.float32)
                self.global_map_input = tf.placeholder(shape=self.max_map_size + (self.map_ch, ), dtype=tf.float32)
                self.visibility_input = tf.placeholder(shape=self.local_map_shape + (1, ), dtype=tf.uint8) if self.use_custom_visibility else None
                local_obj_map_labels = tf.zeros((1, 1, ) + self.local_map_shape + (1, ), dtype=np.uint8)

                self.inference_outputs = train_brain.sequential_inference(
                    self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                    actions, prev_global_map_logodds=self.global_map_input[None],
                    local_obj_maps=local_obj_map_labels,
                    confidence_threshold=self.confidence_threshold,
                    max_confidence=self.max_confidence,
                    max_obj_confidence=0.8,
                    custom_visibility_maps=None if self.visibility_input is None else self.visibility_input[None, None],
                    is_training=True)

                # Add the variable initializer Op.
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            count_number_trainable_params(verbose=True)

            # training session
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            self.sess.run(init)

            load_from_file(self.sess, params.load, partialload=params.partialload, loadcenter=[],
                           skip=params.loadskip, autorename=False)

        self.scan_map_erosion = 5

        self.global_map_logodds = None
        self.xy = None
        self.yaw = None
        self.target_xy = None
        self.step_i = 0
        self.t = time.time()

        self.target_xy_vel = np.zeros((2,))  # TODO remove

        self.reset()


    def __del__(self):
        try:
            self.sess.close()
        except:
            pass

    def reset(self):
        self.global_map_logodds = np.zeros((1, 1) + (self.map_ch, ), np.float32)
        # self.global_map_logodds = np.zeros(self.max_map_size + (2,), np.float32)
        self.step_i = 0
        self.xy = np.zeros((2,), np.float32)
        self.yaw = np.zeros((1,), np.float32)
        self.target_xy = np.zeros((2, ), np.float32)
        self.accumulated_spin = 0.
        self.spin_direction = None

        self.hist1 = np.zeros((0, 3))
        self.hist2 = np.zeros((0, 3))

        print ("Resetting agent.")

    @staticmethod
    def inverse_logodds(x):
        return 1. - 1. / (1 + np.exp(x))

    def plan_and_control(self, xy_mapping, yaw_mapping, lin_vel_sim, ang_vel_sim, target_r_sim, target_fi_sim, global_map_pred, target_xy):

        # Convert to sim representation for expert planning and control
        xy = mapping2sim(xy=xy_mapping)
        yaw = mapping2sim(yaw=yaw_mapping)
        global_map_pred = mapping2sim(global_map_pred=global_map_pred)
        target_xy = mapping2sim(xy=target_xy)
        lin_vel = sim2mapping(lin_vel=lin_vel_sim)
        ang_vel = sim2mapping(ang_vel=ang_vel_sim)

        if self.start_with_spin and np.abs(self.accumulated_spin) < np.deg2rad(360 - 70) and self.step_i < 40:
            if self.spin_direction is None:
                self.spin_direction = -np.sign(target_fi_sim)  # spin opposite direction to the goal
            self.accumulated_spin += ang_vel
            # spin

            print ("%d: spin %f: %f"%(self.step_i, self.spin_direction, self.accumulated_spin))

            action = np.array((0., self.spin_direction * 1.))
            planned_path = np.zeros((0, 2))
            return action, planned_path


        # TODO need to treat case where map needs to be extended !!!!!
        #  Move this logic to map update instead
        # map_mask = 255 - global_map_pred.copy()
        # map_mask[max(int(xy[0]-local_map_max_extent), 0):int(xy[0]+local_map_max_extent),
        #          max(int(xy[1]-local_map_max_extent), 0):int(xy[1]+local_map_max_extent)] = 1.
        # map_mask[max(int(target_map_sim[0]-1.5), 0):int(target_map_sim[0]+2.5),
        #          max(int(target_map_sim[1]-1.5), 0):int(target_map_sim[1]+2.5)] = 1.
        # global_map_pred, offset_2d = GibsonData.reduce_map_size(global_map_pred, map_for_bounding_box=map_mask, margin=8)
        # xy -= offset_2d
        # target_map_sim -= offset_2d
        # #print (global_map_pred.shape)
        # # pdb.set_trace()

        if self.use_object_map:
            assert global_map_pred.shape[-1] == 2
            if IGNORE_OBJECT_MAP:
                global_map_pred = global_map_pred[..., :1]
        else:
            assert global_map_pred.shape[-1] == 1

        # Scan map and cost graph.
        scan_graph, scan_map, resized_scan_map, cost_map = Expert.get_graph_and_eroded_map(
            raw_trav_map=global_map_pred[..., :1],
            trav_map_for_simulator=global_map_pred[..., :1],
            raw_scan_map=global_map_pred,
            rescale_scan_map=1.,
            erosion=self.scan_map_erosion,
            build_graph=False,
            interactive_channel=self.use_object_map and not IGNORE_OBJECT_MAP,
        )

        # TODO split plan and control.
        #  Move it into third file, expert.py, as staticmethods and import that everywhere
        #  Add PID control based on observed lin and ang velocity
        action, obstacle_distance, planned_path, status_message = Expert.policy(
            scan_map=scan_map, scan_graph=scan_graph, pos_map_float=xy, yaw=yaw, target_map_float=target_xy,
            cost_map=cost_map)

        print ("%d: %f %s"%(self.step_i, time.time()-self.t, status_message))
        self.t = time.time()

        planned_path = sim2mapping(xy=planned_path)

        return action, planned_path

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

        if "trav_map" not in observations.keys():
            # No extra info
            observations["trav_map"] = np.zeros((10, 10, 2), np.uint8)
            observations["pose"] = np.zeros((2,))
            observations["yaw"] = np.zeros((1,))
            observations["target"] = np.zeros((2, ))
            observations["rpy"] = np.zeros((3,))
            observations["lin_vel_3d"] = np.zeros((3,))
            observations["ang_vel_3d"] = np.zeros((3,))
            observations["quat"] = np.zeros((4,))
            observations["expert_action"] = np.zeros((2,))


        # Target from extra state obs.
        self.update_pose_from_observation(observations)
        # Updates xy, yaw, target_xy all represented in mapping format  (transposed map used in neural net)

        target_r, target_fi, lin_vel, ang_vel = observations['sensor']

        # Expand map and offset pose if needed, such that target and the surrounding of current pose are all in the map.
        map_shape = self.global_map_logodds.shape
        local_map_max_extent = 110  # TODO need to adjust to local map size and scaler
        target_margin = 8
        min_x = int(min(self.target_xy[0] - target_margin, self.xy[0] - local_map_max_extent) - 1)
        min_y = int(min(self.target_xy[1] - target_margin, self.xy[1] - local_map_max_extent) - 1)
        max_x = int(max(self.target_xy[0] + target_margin, self.xy[0] + local_map_max_extent) + 1)
        max_y = int(max(self.target_xy[1] + target_margin, self.xy[1] + local_map_max_extent) + 1)
        offset_xy = np.array([max(0, -min_x), max(0, -min_y)])
        expand_xy = np.array([max(0, max_x+1-map_shape[0]), max(0, max_y+1-map_shape[1])])
        # print ("Offset", offset_xy, expand_xy)

        self.global_map_logodds = np.pad(
            self.global_map_logodds, [[offset_xy[0], expand_xy[0]], [offset_xy[1], expand_xy[1]], [0, 0]],
            mode='constant', constant_values=0.)
        self.xy += offset_xy
        self.target_xy += offset_xy
        self.true_map_offset_xy += offset_xy
        self.hist1[:, :2] += offset_xy
        self.hist2[:, :2] += offset_xy
        map_shape = self.global_map_logodds.shape

        if map_shape[0] > self.max_map_size[0] or map_shape[1] > self.max_map_size[1]:
            raise ValueError("Required map size is too large. %s > %s"%(str(map_shape), str(self.max_map_size)))

        depth = observations["depth"]
        rgb = observations["rgb"]
        if self.params.mode == 'both':
            images = np.concatenate([depth, rgb], axis=-1)  # these are 0..1  float format
        elif self.params.mode == 'depth':
            images = depth
        else:
            images = rgb
        images = (images * 255).astype(np.uint8)
        images = np.array(images, np.float32)
        # images = images * 255  # to unit8 0..255 format
        images = images * (2. / 255.) - 1.  # to network input -1..1 format

        # TODO mapping label comes from floor_scan_0.png, this might have extra dispersion.
        #  This input is useless since coordinate frames do not match.
        global_map_label = observations["trav_map"]
        global_map_label = global_map_label[:, :, :1]  # input it as uint8
        assert global_map_label.dtype == np.uint8

        # if 'object_states' in observations:
        #     object_map_label = object_map_from_states(observations['object_states'], global_map_label.shape, combine_maps=True)
        #     plt.figure()
        #     plt.imshow(object_map_label[..., 0])
        #     plt.show()
        #     pdb.set_trace()

        global_map_label = sim2mapping(global_map=global_map_label)
        true_map_input = np.zeros(self.max_map_size + (1, ), np.uint8)
        global_map_label = global_map_label[:self.max_map_size[0], :self.max_map_size[1]]
        true_map_input[:global_map_label.shape[0], :global_map_label.shape[1]] = global_map_label

        last_global_map_input = np.zeros(self.max_map_size + (self.map_ch, ), np.float32)
        last_global_map_input[:map_shape[0], :map_shape[1]] = self.global_map_logodds

        feed_dict = {
            self.images_input: images, self.xy_input: self.xy, self.yaw_input: np.array((self.yaw, )),
            self.global_map_input: last_global_map_input,
            self.true_map_input: true_map_input,
        }
        if self.visibility_input is not None:
            visibility_map = ClassicMapping.is_visible_from_depth(depth, self.local_map_shape, zoom_factor=self.brain_requirements.transform_window_scaler)
            feed_dict[self.visibility_input] = visibility_map[:, :, None].astype(np.uint8)

        outputs = self.run_inference(feed_dict)

        global_map_logodds = np.array(outputs.global_map_logodds[0, -1])  # squeeze batch and traj
        global_map_logodds = global_map_logodds[:map_shape[0], :map_shape[1]]
        self.global_map_logodds = global_map_logodds

        local_map_label = outputs.local_map_label[0, 0, :, :, 0]
        local_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 0]

        # global_map_true = self.inverse_logodds(self.global_map_logodds[:, :, 0])
        global_map_true_partial = self.inverse_logodds(self.global_map_logodds[:, :, 0:1])
        if self.use_object_map:
            global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1:3])
            local_obj_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 1]
        else:
            global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1:2])
            local_obj_map_pred = None

        # action = observations["expert_action"]

        if self.map_source == 'true':
            assert global_map_label.ndim == 3
            global_map_for_planning = global_map_label.astype(np.float32) * (1./255.)  # Use full true map
            if self.use_object_map:
                object_map_label = object_map_from_states(observations['object_states'], global_map_label.shape, combine_maps=True)
                assert object_map_label.dtype == np.uint8
                object_map_label = object_map_label.astype(np.float32) * (1./255.)
                global_map_for_planning = np.concatenate((global_map_for_planning, object_map_label), axis=-1)

        # global_map_for_planning = global_map_true_partial
        else:
            global_map_for_planning = global_map_pred

        # threshold
        traversable_threshold = 0.499  # higher than this is traversable
        if IGNORE_OBJECT_MAP:
            object_treshold = 0.  # treat everything as non-object
        else:
            object_treshold = 0.499  # treat as non-object by default
        threshold_const = np.array((traversable_threshold, object_treshold))[None, None, :self.map_ch-1]
        global_map_for_planning = np.array(global_map_for_planning >= threshold_const, np.float32)

        # plan
        action, planned_path = self.plan_and_control(
            self.xy, self.yaw, lin_vel, ang_vel, target_r, target_fi, global_map_pred=global_map_for_planning,
            target_xy=self.target_xy)

        # Visualize agent
        if self.step_i % PLOT_EVERY_N_STEP == 0 and PLOT_EVERY_N_STEP > 0:
            self.visualize_agent(outputs, images, global_map_pred, global_map_for_planning, global_map_label,
                                 global_map_true_partial, local_map_pred, local_map_label, planned_path,
                                 sim_rgb=observations['rgb'], local_obj_map_pred=local_obj_map_pred)

        self.step_i += 1

        return action

    def visualize_agent(self, outputs, images, global_map_pred, global_map_for_planning, global_map_label,
                        global_map_true_partial, local_map_pred, local_map_label, planned_path, sim_rgb=None, local_obj_map_pred=None):
        xy, yaw = self.xy, self.yaw
        status_msg = "step %d" % (self.step_i,)
        visibility_mask = outputs.tiled_visibility_mask[0, 0, :, :, 0]
        # assert global_map_label.shape[-1] == 3
        global_map_label = np.concatenate(
            [global_map_label, np.zeros_like(global_map_label), np.zeros_like(global_map_label)], axis=-1)
        plt.figure("Global map label")
        plt.imshow(global_map_label)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        plt.title(status_msg)
        plt.savefig('./temp/global-map-label.png')
        plt.figure("Global map (%d)" % self.step_i)

        map_to_plot = global_map_pred[..., :1]
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        # plot_target_and_path(target_xy=self.target_xy_vel, path=np.array(self.hist2)[:, :2])
        plt.title(status_msg)
        plt.savefig('./temp/global-map-pred.png')

        if global_map_pred.shape[-1] == 2:
            map_to_plot = global_map_pred[..., 1:2]
            map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
            plt.imshow(map_to_plot)
            plot_viewpoints(xy[0], xy[1], yaw)
            plot_target_and_path(target_xy=self.target_xy, path=planned_path)
            # plot_target_and_path(target_xy=self.target_xy_vel, path=np.array(self.hist2)[:, :2])
            plt.title(status_msg)
            plt.savefig('./temp/global-obj-map-pred.png')

        # plt.figure("Global map true (%d)" % self.step_i)
        # map_to_plot = global_map_true_partial
        # map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        # plt.imshow(map_to_plot)
        # plot_viewpoints(xy[0], xy[1], yaw)
        # plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        # # plot_target_and_path(target_xy=self.target_xy, path=np.array(self.hist1)[:, :2])
        # # plot_target_and_path(target_xy=self.target_xy_vel, path=np.array(self.hist2)[:, :2])
        # plt.title(status_msg)
        # plt.savefig('./temp/global-map-true.png')
        # plt.figure("Global map plan (%d)" % self.step_i)

        map_to_plot = global_map_for_planning
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        plt.title(status_msg)
        plt.savefig('./temp/global-map-plan.png')

        depth, rgb = mapping_visualizer.recover_depth_and_rgb(images)
        if self.params.mode == 'depth' and sim_rgb is not None:
            rgb = sim_rgb
            rgb[:5, :5, :] = 0  # indicate this is not observed

        images_fig, images_axarr = plt.subplots(2, 2, squeeze=True)
        plt.title(status_msg)
        plt.axes(images_axarr[0, 0])
        plt.imshow(depth)
        plt.axes(images_axarr[0, 1])
        plt.imshow(rgb)
        plt.axes(images_axarr[1, 0])
        plt.imshow(local_map_pred * visibility_mask + (1 - visibility_mask) * 0.5)
        plt.axes(images_axarr[1, 1])
        if local_obj_map_pred is not None:
            plt.imshow(local_obj_map_pred * visibility_mask + (1 - visibility_mask) * 0.5)
        else:
            plt.imshow(local_map_label * visibility_mask + (1 - visibility_mask) * 0.5)
        plt.savefig('./temp/inputs.png')
        # pdb.set_trace()
        plt.close('all')

    def run_inference(self, feed_dict):
        outputs = self.sess.run(self.inference_outputs, feed_dict=feed_dict)
        return outputs

    def inverse_rotate(self, vect, r, p, y):
        vect = rotate_vector_3d(vect, 0., 0., -y)
        vect = rotate_vector_3d(vect, 0., -p, 0.)
        vect = rotate_vector_3d(vect, -r, 0., 0.)
        return vect

    def update_pose_from_observation(self, observations):
        map_resolution = 0.05  # this is the agent's map, does not need to match simulator's map
        sim_timestep = 0.1

        target_r, target_fi, lin_vel, ang_vel = observations['sensor']
        target_r = target_r / map_resolution  # convert to map scale
        target_fi = -target_fi
        lin_vel = sim2mapping(lin_vel=lin_vel)
        ang_vel = sim2mapping(ang_vel=ang_vel)

        true_xy = np.copy(observations["pose"][:2])
        true_xy = sim2mapping(xy=true_xy)

        true_target_xy = np.copy(observations["target"][:2])
        true_target_xy = sim2mapping(xy=true_target_xy)

        true_yaw = observations["rpy"][-1]
        true_yaw = sim2mapping(yaw=true_yaw)

        if self.pose_estimation_source == 'true':
            if self.step_i == 0:
                self.true_map_offset_xy = np.zeros((2, ), dtype=np.float32)
                self.true_map_offset_yaw = 0.
                self.target_xy = true_target_xy + self.true_map_offset_xy
                self.yaw = true_yaw
                self.xy = true_xy
                lin_vel = 0.
                ang_vel = 0.
                prev_xy_vel, prev_yaw_vel = true_xy, true_yaw
            else:
                prev_xy_vel = self.hist2[-1][:2]
                prev_yaw_vel = self.hist2[-1][2]

            xy = true_xy + self.true_map_offset_xy
            yaw = true_yaw + self.true_map_offset_yaw

            xy_vel, yaw_vel = self.velocity_update(prev_xy_vel, prev_yaw_vel, lin_vel, ang_vel)
            # xy_vel, yaw_vel = self.velocity_update(prev_xy_vel, yaw, lin_vel, 0.)

            self.xy = xy
            self.yaw = yaw

            self.hist1 = np.append(self.hist1, [[self.xy[0], self.xy[1], self.yaw]], axis=0)
            self.hist2 = np.append(self.hist2, [[xy_vel[0], xy_vel[1], yaw_vel]], axis=0)
            self.target_xy_vel = self.update_target(xy_vel, yaw_vel, target_r, target_fi)

        elif self.pose_estimation_source == 'target':
            if self.step_i == 0:
                # self.target_xy = np.array((0, 0))
                # self.xy = np.array((target_r, 0))
                # self.yaw = -target_fi
                self.target_xy = np.array(true_target_xy)
                self.xy = np.array(true_xy)
                self.yaw = true_yaw
                self.true_map_offset_xy = self.xy - true_xy  # TODO this does not take care of rotation between true and dynamic frames
                self.true_map_offset_yaw = self.yaw - true_yaw

                # vect = self.xy - self.target_xy
                # angle = np.arctan2(vect[1], vect[0])
                # print(np.rad2deg(self.yaw), np.rad2deg(angle), np.rad2deg(target_fi))
                # pdb.set_trace()

            else:
                prev_xy = self.xy
                target_xy = self.target_xy

                prev_vector = prev_xy - target_xy  # pointing from target to xy
                prev_dist = np.linalg.norm(prev_vector)
                prev_angle = np.arctan2(prev_vector[1], prev_vector[0])

                sin_target_fi = np.sin(target_fi)
                sin_psi = sin_target_fi * target_r / prev_dist
                # Need to handle case where target_fi is close to 90 degrees. Not supposed to be getting closer to target without sideway motion.
                # TODO this is not a good way to handle this, it will introduce large error to assume no sideway motion
                if np.abs(sin_target_fi) > np.sin(np.deg2rad(88)):
                    print("Close to 90deg (>88), no side-motion assumption may introduce large error.")

                if np.abs(sin_psi) >= 1.:
                    print ("Negative distance! (fix)")
                    sin_psi = min(1., sin_psi)
                    sin_psi = max(-1., sin_psi)

                psi = np.arcsin(sin_psi)  # sin(target_fi) = sin(np.pi - target_fi) because of symmetry of sin

                if np.abs(target_fi) > np.pi * 0.5:
                    # need to handle -pi..0 range for arcsin
                    psi = np.pi - psi
                angle = prev_angle - psi + target_fi # + np.pi
                xy = np.array((np.cos(angle), np.sin(angle))) * target_r + self.target_xy
                # yaw = prev_angle - psi - np.pi
                yaw = angle + target_fi + np.pi/2

                assert np.isclose(np.linalg.norm(xy - target_xy), target_r)

                # printing
                true_xy += self.true_map_offset_xy
                true_yaw += self.true_map_offset_yaw
                print (self.xy, xy, true_xy, self.yaw, yaw, true_yaw)
                if (np.linalg.norm(xy - self.xy) > 20):
                    print ("large motion")
                    # pdb.set_trace()

                self.xy = xy
                self.yaw = yaw

        elif self.pose_estimation_source == 'velocity':
            if self.step_i == 0:
                self.target_xy = np.array((0, 0))
                self.xy = np.array((target_r, 0))
                self.yaw = target_fi
                # self.target_xy = np.array(true_target_xy)
                # self.xy = np.array(true_xy)
                # self.yaw = true_yaw
                self.true_map_offset_xy = self.xy - true_xy  # TODO this does not take care of rotation between true and dynamic frames
                self.true_map_offset_yaw = self.yaw - true_yaw
            else:
                # First turn than move
                xy, yaw = self.xy, self.yaw

                xy, yaw = self.velocity_update(xy, yaw, lin_vel, ang_vel)

                target_xy = self.update_target(xy, yaw, target_r, target_fi)

                # printing
                true_xy += self.true_map_offset_xy
                true_yaw += self.true_map_offset_yaw
                # print (self.xy, xy, true_xy, self.yaw, yaw, true_yaw, self.target_xy, target_xy, np.linalg.norm(xy-true_xy), np.abs(yaw-true_yaw))

                # if (np.linalg.norm(true_xy - xy) > 20):
                #     print ("large motion")
                #     pdb.set_trace()

                self.xy = xy
                self.yaw = yaw
                self.target_xy = target_xy

        else:
            raise ValueError("Unknown pose estimation source %s"%self.pose_estimation_source)

        return self.xy, self.yaw, self.target_xy

    @staticmethod
    def update_target(xy, yaw, target_r, target_fi):
        # update target as well
        vect = np.array((-np.cos(yaw - target_fi - np.pi / 2), -np.sin(yaw - target_fi - np.pi / 2))) * target_r
        target_xy = xy + vect
        return target_xy

    @staticmethod
    def velocity_update(xy, yaw, lin_vel, ang_vel):
        yaw += ang_vel
        vect = np.array((-np.cos(yaw - np.pi/2), -np.sin(yaw - np.pi/2))) * lin_vel
        xy += vect

        return xy, yaw

    def debug_pose(self, observations, xy, yaw):
        # Main issue is that base_env does not pass computeForwardDinamics to pybullet, so the pose and orientation
        # readings are false (lagging behind? sometimes with pyhsics_step=0.1 after p.step() pose is not updated at all)
        # These incorrect rpy is used to transform the correct lin_vel and ang_vel
        # But image seems to be rendered from the incorrect pose as well
        # Meaning image is consistant with poses, but inconsistent with velocity readings
        # Actual robot dynamics is consistent (?) with velocities but not with pose readings

        xy = np.copy(observations["pose"][:2])
        yaw = observations["rpy"][2:3]
        xy = sim2mapping(xy=xy)
        yaw = sim2mapping(yaw=np.array(yaw))

        target_r, target_fi, lin_vel, ang_vel = observations['sensor']
        lin_vel_3d = np.copy(observations['lin_vel_3d'])
        ang_vel_3d = np.copy(observations['ang_vel_3d'])
        quat = np.copy(observations['quat'])
        pos_world = np.copy(observations['pose'])

        sim_map_resolution = 0.05
        sim_timestep = 0.1
        sim_2_mapping_scaler = 1. / sim_map_resolution
        target_r *= sim_2_mapping_scaler
        lin_vel *= sim_2_mapping_scaler
        lin_vel *= sim_timestep
        ang_vel *= sim_timestep
        lin_vel_3d *= sim_timestep
        ang_vel_3d *= sim_timestep
        rel_target_xy = np.array([np.cos(target_fi), np.sin(target_fi)]) * target_r
        abs_target_xy = sim2mapping(xy=observations['target'])
        abs_target_vect = abs_target_xy - xy
        true_target_r = np.sqrt(np.sum(np.square(abs_target_vect)))
        abs_yaw = np.arctan2(abs_target_vect[1], abs_target_vect[0]) - target_fi - np.pi / 2
        yaw = np.squeeze(yaw)
        rpy = observations['rpy']
        projected_rpy = rotate_vector_3d(np.array([0., 0., yaw]), *(rpy))
        projected_yaw = projected_rpy[2]
        if self.step_i == 0:
            self.target_xy = rel_target_xy + self.xy
            if not (np.isclose(lin_vel, 0) and np.isclose(ang_vel, 0.)):
                print("Already moving at reset: %f %f" % (lin_vel, ang_vel))

            self.prev_lin_vel = lin_vel
            self.prev_ang_vel = ang_vel
            self.prev_env_xy = xy
            self.prev_env_yaw = yaw
            self.prev_rpy = np.copy(rpy)
            self.prev_quat = np.copy(quat)
            self.prev_pos_world = np.copy(pos_world)
            self.prev_projected_yaw = projected_yaw
            self.accumulated_yaw = 0
            self.initial_projected_yaw = projected_yaw
            self.initial_yaw = yaw

        # TODO the order of rpy matters.

        rpy_vel = rpy - self.prev_rpy
        projected_rpy_vel = rotate_vector_3d(rpy_vel, *rpy)
        # projected_ang_vel = rotate_vector_3d(rpy_vel, *(rpy))[2]
        projected_ang_vel = projected_yaw - self.prev_projected_yaw
        true_lin_vel = np.sqrt(np.sum(np.square(xy - self.prev_env_xy)))
        true_ang_vel = yaw - self.prev_env_yaw
        self.accumulated_yaw += ang_vel
        projected_yaw_error = projected_yaw - self.accumulated_yaw - self.initial_projected_yaw
        yaw_error = yaw - self.accumulated_yaw - self.initial_yaw
        avg_lin_vel = (self.prev_lin_vel + lin_vel) * 0.5
        avg_ang_vel = (self.prev_ang_vel + ang_vel) * 0.5
        # TODO does not match, but maybe mine is wrong (!) because i only take yaw from rpy, not transform it into horizontal plance
        # print (self.prev_env_xy)
        # print (xy)

        correct_lin_vel = np.sqrt(np.square(lin_vel_3d[1]) + np.square(lin_vel_3d[0])) * sim_2_mapping_scaler
        correct_ang_vel = ang_vel_3d[-1]

        recovered_lin_vel_3d = self.inverse_rotate(np.array([lin_vel, 0., 0.]), *rpy)
        recovered_lin_vel = np.sqrt(np.square(recovered_lin_vel_3d[1]) + np.square(recovered_lin_vel_3d[0]))

        print(yaw, projected_yaw)
        print(true_lin_vel, true_ang_vel, avg_lin_vel, avg_ang_vel, np.abs(rpy[0]) + np.abs(rpy[1]))
        print(true_lin_vel / avg_lin_vel, true_ang_vel / avg_ang_vel)
        print(true_lin_vel / lin_vel, true_ang_vel / ang_vel)
        print(projected_ang_vel / ang_vel, projected_ang_vel / avg_ang_vel)
        print("Absolute error %f; projected %f target-based %f; target-based-projected %f" % (
        yaw_error, projected_yaw_error, yaw - abs_yaw, projected_yaw - abs_yaw))
        print("Target r: %f %f diff = %f" % (true_target_r, target_r, true_target_r - target_r))
        print ("Derived from 3d velocities", true_lin_vel / correct_lin_vel, true_ang_vel/correct_ang_vel)
        print("Recover from observed velocities with true rpy", true_lin_vel / recovered_lin_vel, true_ang_vel / correct_ang_vel)

        coord1_ang_vel = self.inverse_rotate(ang_vel_3d, *self.prev_rpy)
        coord2_ang_vel = self.inverse_rotate(ang_vel_3d, *rpy)

        pdb.set_trace()

        self.prev_lin_vel = np.copy(lin_vel)
        self.prev_ang_vel = np.copy(ang_vel)
        self.prev_env_xy = np.copy(xy)
        self.prev_env_yaw = np.copy(yaw)
        self.prev_rpy = np.copy(rpy)
        self.prev_quat = np.copy(quat)
        self.prev_pos_world = np.copy(pos_world)
        self.prev_projected_yaw = np.copy(projected_yaw)
        yaw = np.array((yaw,))


def get_agent(agent_class, ckpt_path=""):
    return ExpertAgent()

    if agent_class == "Random":
        return RandomAgent()
    elif agent_class == "ForwardOnly":
        return ForwardOnlyAgent()
    elif agent_class == "SAC":
        return SACAgent(root_dir=ckpt_path)
