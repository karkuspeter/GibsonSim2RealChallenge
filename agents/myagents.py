from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from agents.simple_agent import RandomAgent, ForwardOnlyAgent
from agents.rl_agent import SACAgent

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers
import matplotlib.animation as animation
import os
import time

from train import get_brain
from common_net import load_from_file, count_number_trainable_params
from preprocess import sim2mapping, mapping2sim, object_map_from_states
from visualize_mapping import plot_viewpoints, plot_target_and_path, mapping_visualizer
from multiprocessing import Process, Queue
import queue

import numpy as np
import tensorflow as tf

from gibsonagents.expert import Expert
from gibsonagents.classic_mapping import ClassicMapping, rotate_2d
from gibsonagents.pathplanners import Dstar_planner

# from gibson2.core.physics.scene import BuildingScene
from gibson2.envs.locomotor_env import NavigateEnv
from preprocess import GibsonData
from gibson2.utils.utils import rotate_vector_3d

try:
    import ipdb as pdb
except Exception:
    import pdb


MAP_SOURCE = 'pred'  # 'pred'   # 'label'  # 'pred'
USE_OBJECT_MAP_FOR_TRACK = [False, True, True]
IGNORE_OBJECT_MAP_FOR_COST = True
USE_DYNAMIC_MAP = False
CLEAR_DYNAMIC_MAP_EACH_STEP = False
# DYNAMIC_MAP_CLEAR_DISTANCE = 10  # clear top n rows of dynamic map - need to implement within mapper brain
DYNAMIC_MAP_RESCALE = 1.  # the smaller the more confident the prediction 0.5-->0.05
LINVEL_DIRECTIONAL = True
USE_GPU = True

SAVE_VIDEO = 10
PLOT_EVERY_N_STEP = -1  # 10

# sim eval
POSE_ESTIMATION_SOURCE = 'true'  # 'velocity'  #''true' # 'velocity'
ARTIFICIAL_DELAY_STEPS = 0  #3  # 3
MOTION_PARAMS = 'sim'  # 'sim'  #'real'  #'real'
MAX_PLAN_SECONDS = 0.15  # 0.15  # 0.065  # 0.075  # 0.075
PLANNER_KEEP_ALIVE = 50   # DISABLE THIS FOR REAL TRACK

# # real track submission
# POSE_ESTIMATION_SOURCE = 'velocity'  #''true' # 'velocity'
# ARTIFICIAL_DELAY_STEPS = 0  #3  # 3
# MOTION_PARAMS = 'real'  # 'sim'  #'real'  #'real'
# MAX_PLAN_SECONDS = 0.065  # 0.15  # 0.065  # 0.075  # 0.075
# PLANNER_KEEP_ALIVE = -1  # 50   # DISABLE THIS FOR REAL TRACK

TRAVERSABLE_THRESHOLD = [0.38, 0.38, 0.499]  # 0.499  moved to params
OBSTACLE_DOWNWEIGHT = [False, True, True]
OBSTACLE_DOWNWEIGHT_DISTANCE = 20  # from top, smaller the further
OBSTACLE_DOWNWEIGHT_SCALARS = (0.3, 0.8) # (0.3, 0.8)

START_WITH_SPIN = True
MAX_SPIN_STEPS = 40
REDUCE_MAP_FOR_PLANNING = True
SAFETY_REPLAN_IF_COST_INCREASE = 40

# %run -p -s cumulative  -l 80 -T temp/prun4 agent.py  --gibson_mode evalsubmission --gibson_split evaltest


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


def planner_process_function(input_queue, output_queue):
    while True:
        if PLANNER_KEEP_ALIVE > 0:
            try:
                planid, offset, args, kwargs = input_queue.get(timeout=PLANNER_KEEP_ALIVE)
            except queue.Empty:
                print ("Exiting planner process.")
                return
        else:
            planid, offset, args, kwargs = input_queue.get()
        outputs = Expert.plan_policy(*args, **kwargs)

        # Clear earlier results from queue
        try:
            output_queue.get(False)
        except queue.Empty:
            pass
        output_queue.put((planid, offset, outputs))


class MappingAgent(RandomAgent):
    def __init__(self, params, logdir='./temp/', scene_name='scene'):
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
        params.traversable_threshold = TRAVERSABLE_THRESHOLD[track_i]

        print (params)

        assert params.sim == 'gibson'

        self.planner_input_queue = None
        self.planenr_output_queue = None
        self.last_planner_output = None
        self.last_path_cost = 0.
        self.planner_process = None
        self.plan_id = -1
        self.plan_offset = np.array((0, 0), np.int32)

        self.summary_str = ""
        self.filename_addition = ""
        self.logdir = logdir
        self.scene_name = scene_name
        self.num_videos_saved = 0

        # self.pathplanner = Dstar_planner(single_thread=False)

        if not USE_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.params = params
        self.pose_estimation_source = POSE_ESTIMATION_SOURCE
        self.map_source = MAP_SOURCE
        self.start_with_spin = START_WITH_SPIN
        self.max_confidence = 0.96   # 0.98
        self.confidence_threshold = None  # (0.2, 0.01)  # (0.35, 0.05)
        self.use_custom_visibility = (self.params.visibility_mask in [2, 20, 21])
        self.use_object_map = (self.params.object_map > 0)  # if want to change also need to overwrite params.object_map

        self.map_ch = (3 if self.use_object_map else 2)
        self.accumulated_spin = 0.
        self.spin_direction = None

        self.delay_buffer = []
        self.last_expert_state = None

        params.batchdata = 1
        params.trajlen = 1
        sensor_ch = (1 if params.mode == 'depth' else (3 if params.mode == 'rgb' else 4))
        self.max_map_size = (1200, 1200)
        self.plan_map_size = (360, 360) if REDUCE_MAP_FOR_PLANNING else self.max_map_size

        # Build brain
        with tf.Graph().as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
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

                if OBSTACLE_DOWNWEIGHT[track_i]:
                    custom_obstacle_prediction_weight = Expert.get_obstacle_prediction_weight(OBSTACLE_DOWNWEIGHT_DISTANCE, OBSTACLE_DOWNWEIGHT_SCALARS, self.local_map_shape)
                else:
                    custom_obstacle_prediction_weight = None

                self.inference_outputs = train_brain.sequential_inference(
                    self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                    actions, prev_global_map_logodds=self.global_map_input[None],
                    local_obj_maps=local_obj_map_labels,
                    confidence_threshold=self.confidence_threshold,
                    max_confidence=self.max_confidence,
                    max_obj_confidence=0.8,
                    custom_visibility_maps=None if self.visibility_input is None else self.visibility_input[None, None],
                    custom_obstacle_prediction_weight=custom_obstacle_prediction_weight,
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

        self.frame_traj_data = []
        self.episode_i = -1

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
        self.last_expert_state = None

        self.hist1 = np.zeros((0, 3))
        self.hist2 = np.zeros((0, 3))

        self.delay_buffer = [np.zeros((2,), np.float32)] * ARTIFICIAL_DELAY_STEPS
        self.past_actions = [np.zeros((2,), np.float32)] * self.params.motion_delay_steps
        self.traj_xy = []
        self.traj_yaw = []
        self.pred_traj_n_xy = []
        self.pred_traj_n_yaw = []
        self.pred_errors = []

        self.plan_id += 1
        self.plan_offset = np.array((0, 0), np.int32)

        # Launch new planner process
        self.start_planner_process_if_needed()

        try:
            while True:
                self.planner_input_queue.get(False)
        except queue.Empty:
            pass
        try:
            while True:
                self.planner_output_queue.get(False)
        except queue.Empty:
            pass
        # while not self.planner_input_queue.empty():
        #     self.planner_input_queue.get()
        # while not self.planner_output_queue.empty():
        #     self.planner_output_queue.get()

        # # Launch new planner process
        # if self.planner_process is not None:
        #     self.planner_process.terminate()
        # self.planner_input_queue = Queue()
        # self.planner_output_queue = Queue()

        # self.pathplanner.reset()

        self.last_planner_output = None
        self.last_path_cost = 0.
        self.last_plan_update_step = -1

        self.episode_i += 1

        self.reset_video_writer()

        print ("Resetting agent.")

    def start_planner_process_if_needed(self):
        if (self.planner_process is None) or (not self.planner_process.is_alive()):
            self.planner_input_queue = Queue()
            self.planner_output_queue = Queue()
            self.planner_process = Process(target=planner_process_function,
                                           args=(self.planner_input_queue, self.planner_output_queue))
            self.planner_process.start()

    @staticmethod
    def inverse_logodds(x):
        return 1. - 1. / (1 + np.exp(x))

    def motion_prediction(self, xy, yaw, past_xy, past_yaw, num_steps, model='lin2'):

        assert len(self.past_actions) >= num_steps
        xy_traj = [xy]
        yaw_traj = [yaw]

        if num_steps > 0:
            past_n_actions = np.stack(self.past_actions[-num_steps:], axis=0)
            if len(past_xy) < 1:
                lin_vel = 0.
                ang_vel = 0.
            else:
                motion_vect = xy - past_xy[-1]
                if LINVEL_DIRECTIONAL:
                    motion_vect = np.array((-np.cos(yaw - np.pi / 2), -np.sin(yaw - np.pi / 2))) * motion_vect
                    # print(motion_vect, past_xy[-1], xy, yaw)
                    lin_vel = np.linalg.norm(motion_vect, axis=0) * np.sign(motion_vect[1])
                else:
                    lin_vel = np.linalg.norm(motion_vect, axis=0)
                ang_vel = yaw - past_yaw[-1]
                ang_vel = (ang_vel + np.pi) % (2*np.pi) - np.pi

            # Unroll motion model with past n actions
            for act_fwd, act_rot in past_n_actions:
                # predict lin and rot velocities
                if model == 'lin1':
                    # From real log (static)
                    # lin_scaler = 0.6540917264778654
                    # ang_scaler = 0.07866921965909571
                    # # Sym v2
                    # lin_scaler = 1.3600789433885767
                    # ang_scaler = 0.1287800474224603

                    if MOTION_PARAMS == 'sim':
                        # ./temp/evals/eval_static_06-04-15-34-17_out.log
                        lin_scaler, ang_scaler = 0.7093248690037459, 0.1224500742553487
                    elif MOTION_PARAMS == 'inflated':
                        # ./temp/evals/eval_static_06-04-15-34-17_out.log
                        lin_scaler, ang_scaler = 0.7093248690037459 * 1.2, 0.1224500742553487 * 1.2
                    else:
                        lin_scaler, ang_scaler = 0.6395001626915873, 0.08237032666912558

                    pred_lin_vel = act_fwd * lin_scaler
                    pred_rot_vel = act_rot * ang_scaler
                elif model == 'lin2':
                    # ./temp/evals/eval_static_06-04-15-34-17_out.log
                    param_vect = np.array((0.5474649003027493, 0.39226761787103714, 0.5027164023455273, 0.07327970777300274))

                    pred_lin_vel = param_vect[0] * lin_vel + param_vect[1] * act_fwd
                    pred_rot_vel = param_vect[2] * ang_vel + param_vect[3] * act_rot
                else:
                    raise ValueError(model)

                # use predicted velocity to update pose
                xy, yaw = self.velocity_update(xy, yaw, pred_lin_vel, pred_rot_vel)
                xy_traj.append(xy)
                yaw_traj.append(yaw)

        return xy_traj, yaw_traj

    @staticmethod
    def update_planned_path_with_offset_change(path_map, offset, current_offset):
        return path_map + (current_offset - offset)[None]

    def plan_and_control(self, xy_mapping, yaw_mapping, prev_yaw_mapping, target_fi_sim, # lin_vel_sim, ang_vel_sim, target_r_sim, target_fi_sim,
                         global_map_pred, target_xy):
        # Convert to sim representation for expert planning and control
        xy = mapping2sim(xy=xy_mapping)
        yaw = mapping2sim(yaw=yaw_mapping)
        prev_yaw = mapping2sim(yaw=prev_yaw_mapping)
        global_map_pred = mapping2sim(global_map_pred=global_map_pred)
        target_xy = mapping2sim(xy=target_xy)
        current_plan_offset = mapping2sim(xy=self.plan_offset)
        # lin_vel = sim2mapping(lin_vel=lin_vel_sim)
        # ang_vel = sim2mapping(ang_vel=ang_vel_sim)
        #
        # if self.start_with_spin and np.abs(self.accumulated_spin) < np.deg2rad(360 - 70) and self.step_i < MAX_SPIN_STEPS:
        #     if self.spin_direction is None:
        #         self.spin_direction = np.sign(target_fi_sim)  # spin opposite direction to the goal   # was negative when using ang_vel
        #     self.accumulated_spin += yaw-prev_yaw
        #     # spin
        #
        #     status_message = ("spin %f: %f"%(self.spin_direction, self.accumulated_spin))
        #
        #     action = np.array((0., self.spin_direction * 1.))
        #     planned_path = np.zeros((0, 2))
        #     return action, planned_path, sim2mapping(xy=xy.copy()), status_message
        #

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
            if IGNORE_OBJECT_MAP_FOR_COST:
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
            interactive_channel=self.use_object_map and not IGNORE_OBJECT_MAP_FOR_COST,
        )

        # # Policy in one step
        # action, obstacle_distance, planned_path, status_message, self.last_expert_state = Expert.policy(
        #     scan_map=scan_map, scan_graph=scan_graph, pos_map_float=xy, yaw=yaw, target_map_float=target_xy,
        #     cost_map=cost_map, prev_state=self.last_expert_state)

        # # planner
        # path_map, path_len_map = Expert.plan_policy(
        #     scan_graph=scan_graph, pos_map_float=xy, yaw=yaw, target_map_float=target_xy,
        #     cost_map=cost_map)

        # planner in separate process
        args = ()
        kwargs = dict(scan_graph=scan_graph, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=cost_map,
                      max_map_size=self.plan_map_size)

        self.start_planner_process_if_needed()

        # clear queue in case last input was not even taken
        try:
            while True:
                self.planner_input_queue.get(False)
        except queue.Empty:
            pass
        try:
            # Also clear output
            while True:
                plan_id, plan_offset, (path_map, path_len_map) = self.planner_output_queue.get(False)
                # update plan if one already existed
                if self.last_planner_output is not None and plan_id == self.plan_id:
                    path_map = self.update_planned_path_with_offset_change(path_map, plan_offset, current_plan_offset)
                    self.last_planner_output = (np.array(path_map), path_len_map)
        except queue.Empty:
            pass
        self.planner_input_queue.put((self.plan_id, current_plan_offset, args, kwargs))
        try:
            plan_id, plan_offset, (path_map, path_len_map) = self.planner_output_queue.get(timeout=MAX_PLAN_SECONDS)
            if plan_id != self.plan_id:
                raise queue.Empty()
            self.last_plan_update_step = self.step_i
            path_map = self.update_planned_path_with_offset_change(path_map, plan_offset, current_plan_offset)
        except queue.Empty:
            if self.last_planner_output is None:
                print ("Plan is not ready, but we dont have any plan yet..")
                while True:
                    plan_id, plan_offset, (path_map, path_len_map) = self.planner_output_queue.get(timeout=30.)
                    if plan_id == self.plan_id:
                        break
                    print ("Wrong plan from earlier episode")
                path_map = self.update_planned_path_with_offset_change(path_map, plan_offset, current_plan_offset)
                self.last_plan_update_step = self.step_i
                path_cost = np.sum(cost_map[path_map[:, 0], path_map[:, 1]])
                self.last_path_cost = path_cost
            else:
                # find the closest along the path and skip anything before that
                path_map, path_len_map = self.last_planner_output
                assert path_map.ndim == 2
                path_cost = np.sum(cost_map[path_map[:, 0], path_map[:, 1]])
                cost_increase = path_cost - self.last_path_cost

                closest_i = np.argmin(np.linalg.norm(path_map - xy[None], axis=1))
                closest_i = max(min(closest_i, len(path_map)-2), 0)  # do not cut shorter than the last 2 steps (or single step if only has one step)
                path_map = path_map[closest_i:]

                print ("Plan is not ready. Cost increase %.1f Using earlier from step %d %d | path %d / %d."%(
                    cost_increase, self.step_i-self.last_plan_update_step, self.last_plan_update_step, closest_i, len(path_map)))

                if cost_increase > SAFETY_REPLAN_IF_COST_INCREASE and self.step_i < 300:
                    action = np.array((0., 0.), np.float32)
                    status_message = "Waiting for updated plan, cost increase too large."
                    return action, sim2mapping(xy=path_map), sim2mapping(xy=xy.copy()), status_message


        # # DStar planner
        # path_map, path_len_map = self.pathplanner.dstar_path(
        #     cost_map, tuple(xy.astype(np.int32)), tuple(target_xy.astype(np.int32)), timeout=MAX_PLAN_SECONDS, strict_timeout=(self.last_planner_output is not None))
        # if path_map is None:
        #     # find the closest along the path and skip anything before that
        #     path_map, path_len_map = self.last_planner_output
        #     assert path_map.ndim == 2
        #     closest_i = np.argmin(np.linalg.norm(path_map - xy[None], axis=1))
        #     closest_i = max(min(closest_i, len(path_map)-2), 0)  # do not cut shorter than the last 2 steps (or single step if only has one step)
        #     print ("Plan is not ready. Using earlier one, from step %d / %d."%(closest_i, len(path_map)))
        #     path_map = path_map[closest_i:]

        self.last_planner_output = (np.array(path_map), path_len_map)

        # controller
        #  TODO Add PID control based on observed lin and ang velocity
        action, obstacle_distance, planned_path, subgoal, status_message, self.last_expert_state =  Expert.control_policy(
            path_map, path_len_map, scan_map=scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy,
            cost_map=cost_map, prev_state=self.last_expert_state, control_params=self.params.pid_params)

        # FOR VIDEO ONLY OVERWRITE HERE
        if self.start_with_spin and np.abs(self.accumulated_spin) < np.deg2rad(360 - 70) and self.step_i < MAX_SPIN_STEPS:
            if self.spin_direction is None:
                self.spin_direction = np.sign(target_fi_sim)  # spin opposite direction to the goal   # was negative when using ang_vel
            self.accumulated_spin += yaw-prev_yaw
            # spin

            status_message = ("spin %f: %f"%(self.spin_direction, self.accumulated_spin))
            action = np.array((0., self.spin_direction * 1.))

        return action, sim2mapping(xy=planned_path), sim2mapping(xy=subgoal), status_message

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
            observations["object_states"] = np.zeros((0, 4))
            observations["expert_action"] = np.zeros((2,))


        # Target from extra state obs.
        prev_yaw = self.yaw
        self.update_pose_from_observation(observations)
        # Updates xy, yaw, target_xy all represented in mapping format  (transposed map used in neural net)

        target_r, target_fi, lin_vel, ang_vel = observations['sensor']

        # # Motion prediction
        pred_xy_traj, pred_yaw_traj = self.motion_prediction(
            self.xy, self.yaw, self.traj_xy, self.traj_yaw, num_steps=self.params.motion_delay_steps,
            model=self.params.motion_model)
        xy_for_planning = pred_xy_traj[-1]
        yaw_for_planning = pred_yaw_traj[-1]
        if len(pred_yaw_traj) >= 2:
            prev_yaw_for_planning = pred_yaw_traj[-2]
        else:
            prev_yaw_for_planning = prev_yaw if self.step_i > 0 else self.yaw

        # Expand map and offset pose if needed, such that target and the surrounding of current pose are all in the map.
        map_shape = self.global_map_logodds.shape
        local_map_max_extent = 110  # TODO need to adjust to local map size and scaler
        target_margin = 8
        min_xy = np.stack([self.target_xy - target_margin, self.xy - local_map_max_extent, xy_for_planning - local_map_max_extent], axis=0)
        min_xy = np.min(min_xy, axis=0).astype(np.int) - 1
        max_xy = np.stack([self.target_xy + target_margin, self.xy + local_map_max_extent, xy_for_planning + local_map_max_extent], axis=0)
        max_xy = np.max(max_xy, axis=0).astype(np.int) + 1
        offset_xy = np.array([max(0, -min_xy[0]), max(0, -min_xy[1])])
        expand_xy = np.array([max(0, max_xy[0]+1-map_shape[0]), max(0, max_xy[1]+1-map_shape[1])])
        # print ("Offset", offset_xy, expand_xy)

        self.global_map_logodds = np.pad(
            self.global_map_logodds, [[offset_xy[0], expand_xy[0]], [offset_xy[1], expand_xy[1]], [0, 0]],
            mode='constant', constant_values=0.)
        if np.any(offset_xy > 0):
            self.xy += offset_xy
            xy_for_planning += offset_xy
            self.traj_xy = [val + offset_xy for val in self.traj_xy]
            self.pred_traj_n_xy = [val + offset_xy[None] for val in self.pred_traj_n_xy]
            self.target_xy += offset_xy
            self.true_map_offset_xy += offset_xy
            self.hist1[:, :2] += offset_xy
            self.hist2[:, :2] += offset_xy
            self.plan_offset += offset_xy
            if self.last_planner_output is not None:
                self.last_planner_output = (self.last_planner_output[0] + offset_xy[None], self.last_planner_output[1])
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
        #  This input is useless since coordinate frames do not match.  !!!
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
            visibility_map = ClassicMapping.is_visible_from_depth(depth, self.local_map_shape, sim=self.params.sim, zoom_factor=self.brain_requirements.transform_window_scaler)
            feed_dict[self.visibility_input] = visibility_map[:, :, None].astype(np.uint8)

        outputs = self.run_inference(feed_dict)

        global_map_logodds = np.array(outputs.global_map_logodds[0, -1])  # squeeze batch and traj
        global_map_logodds = global_map_logodds[:map_shape[0], :map_shape[1]]
        self.global_map_logodds = global_map_logodds

        local_map_label = outputs.local_map_label[0, 0, :, :, 0]
        local_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 0]

        # global_map_true = self.inverse_logodds(self.global_map_logodds[:, :, 0])
        global_map_true_partial = self.inverse_logodds(self.global_map_logodds[:, :, 0:1])
        if self.use_object_map and USE_DYNAMIC_MAP and self.sim2real_track == 'dynamic':
            # Special treatment of temporary dynamic map.
            global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1:2])
            dynamic_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 2:3])

            # Merge dynamic map into standard map
            dynamic_object_filter = (dynamic_map_pred <= 0.499)  # predict presence of dynamic object
            # Replace global map prediction where dynamic map prediction is more pessimistic
            global_map_pred[dynamic_object_filter] = np.minimum(global_map_pred[dynamic_object_filter], dynamic_map_pred[dynamic_object_filter] * DYNAMIC_MAP_RESCALE)
            print ("dynamic map %d"%np.count_nonzero(dynamic_object_filter))

            # Clear dynamic map for next step
            if CLEAR_DYNAMIC_MAP_EACH_STEP:
                self.global_map_logodds[:, :, 2:3] = 0.

            local_obj_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 1]
        elif self.use_object_map:
            global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1:3])
            local_obj_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 1]
        else:
            global_map_pred = self.inverse_logodds(self.global_map_logodds[:, :, 1:2])
            local_obj_map_pred = None

        # action = observations["expert_action"]

        true_global_map = global_map_label.astype(np.float32) * (1./255.)  # Use full true map
        if self.use_object_map:
            object_map_label = object_map_from_states(observations['object_states'], global_map_label.shape, combine_maps=True)
            assert object_map_label.dtype == np.uint8
            object_map_label = object_map_label.astype(np.float32) * (1./255.)
            true_global_map = np.concatenate((true_global_map, object_map_label), axis=-1)
        true_global_map = np.pad(
            true_global_map, [[self.plan_offset[0], 0], [self.plan_offset[1], 0], [0, 0]],
            mode='constant', constant_values=0.5)

        if self.map_source == 'true':
            assert POSE_ESTIMATION_SOURCE == 'true'  # otherwise need to match coordinate frames
            assert global_map_label.ndim == 3
            global_map_for_planning = true_global_map
        else:
            global_map_for_planning = global_map_pred

        # threshold
        traversable_threshold = self.params.traversable_threshold
        if IGNORE_OBJECT_MAP_FOR_COST:
            object_treshold = 0.  # treat everything as non-object
        else:
            object_treshold = traversable_threshold  # treat as non-object by default
        threshold_const = np.array((traversable_threshold, object_treshold))[None, None, :self.map_ch-1]
        binary_global_map_for_planning = np.array(global_map_for_planning >= threshold_const, np.float32)

        # plan
        action, planned_path, planned_subgoal, status_message = self.plan_and_control(
            xy_for_planning, yaw_for_planning, prev_yaw_for_planning, # lin_vel, ang_vel, target_r,
            target_fi,
            global_map_pred=binary_global_map_for_planning,
            target_xy=self.target_xy)


        target_status = "%4d %2.1f"%(int(np.rad2deg(target_fi)),  target_r)
        control_status = ("%d: %f %s"%(self.step_i, time.time()-self.t, status_message))
        print (control_status + " | " + target_status)
        self.t = time.time()

        # Visualize agent
        if self.step_i % PLOT_EVERY_N_STEP == 0 and PLOT_EVERY_N_STEP > 0:
            self.visualize_agent(outputs, images, global_map_pred, binary_global_map_for_planning, global_map_label,
                                 global_map_true_partial, local_map_pred, local_map_label, planned_path,
                                 sim_rgb=observations['rgb'], local_obj_map_pred=local_obj_map_pred)

        self.past_actions.append(np.array(action))
        self.traj_xy.append(self.xy.copy())
        self.traj_yaw.append(self.yaw.copy())
        self.pred_traj_n_xy.append(np.array(pred_xy_traj))
        self.pred_traj_n_yaw.append(np.array(pred_yaw_traj))

        pred_errors = []
        for i in range(1, self.params.motion_delay_steps+1):
            if len(self.pred_traj_n_xy) <= i:
                error_xy = 0.
                error_yaw = 0.
            else:
                error_xy = np.linalg.norm(self.pred_traj_n_xy[-i-1][i] - self.xy, axis=0)
                error_yaw = np.abs(self.pred_traj_n_yaw[-i-1][i] - self.yaw)
            pred_errors.append((error_xy, error_yaw))
        self.pred_errors.append(pred_errors)
        motion_pred_status = (" ".join(["p%d: %.1f,"%(i+1, pred_errors[i][0]) for i in range(self.params.motion_delay_steps)]) + " | " +
               " ".join(["p%d: %.1f,"%(i+1, np.rad2deg(pred_errors[i][1])) for i in range(self.params.motion_delay_steps)]))
        print (motion_pred_status)

        # Add artificial delay
        if ARTIFICIAL_DELAY_STEPS > 0:
            assert len(self.delay_buffer) == ARTIFICIAL_DELAY_STEPS
            self.delay_buffer.append(action)
            action = self.delay_buffer[0]
            self.delay_buffer = self.delay_buffer[1:]
            intended_action = self.delay_buffer[-1]
        else:
            intended_action = action

        act_status = ("%d: delay%d %.3f %.3f %.3f act %.2f %.2f --> %.3f %.3f %.3f act %f %f "%(
            self.step_i, ARTIFICIAL_DELAY_STEPS, self.xy[0], self.xy[1], np.rad2deg(self.yaw),
            intended_action[0], intended_action[1],
            pred_xy_traj[-1][0], pred_xy_traj[-1][1], np.rad2deg(pred_yaw_traj[-1]), action[0], action[1]))
        print (act_status)

        if SAVE_VIDEO > self.num_videos_saved:
            self.frame_traj_data.append(dict(
                rgb=observations['rgb'], global_map=global_map_pred.copy(),
                true_global_map=true_global_map.copy(), xy=self.xy.copy(), yaw=self.yaw.copy(),
                target_xy=self.target_xy.copy(),
                path=planned_path.copy(), subgoal=planned_subgoal.copy(),
                target_status=target_status, control_status=control_status, act_status=act_status))

        self.step_i += 1
        return action

    def video_update(self, frame_i):
        # frame skip of 3
        if frame_i % 3 == 0:
            ind = min(frame_i // 3, len(self.frame_traj_data)-1)
            self.video_image_ax.set_data(self.frame_traj_data[ind]['rgb'])
            self.video_text_ax1.set_text(self.frame_traj_data[ind]['target_status'])
            split_str = self.frame_traj_data[ind]['control_status']
            # Attempt to break lines
            segs = split_str.split("[")
            if len(segs) > 1:
                split_str = segs[0] + "\n["+"[".join(segs[1:])
            segs = split_str.split(" v=")
            if len(segs) > 1:
                split_str = segs[0] + "\nv=" + " v=".join(segs[1:])
            self.video_text_ax2.set_text(split_str)

            if self.video_global_map_ax is not None:
                xy = self.frame_traj_data[ind]['xy']
                target_xy = self.frame_traj_data[ind]['target_xy']
                subgoal = self.frame_traj_data[ind]['subgoal']
                path = self.frame_traj_data[ind]['path']
                if len(path) == 0:
                    path = xy[None]

                global_map = np.atleast_3d(self.frame_traj_data[ind]['global_map'])
                global_map = np.tile(global_map[:, :, :1], [1, 1, 3])
                true_map = np.atleast_3d(self.frame_traj_data[ind]['true_global_map'])

                # # Crop true map to meaningful area
                # true_map = true_map[:np.max(np.nonzero(true_map)[0])+2, :np.max(np.nonzero(true_map)[1])+2, :1]
                # true_map = np.tile(true_map, [1, 1, 3])
                # true_map = np.pad(true_map, [[0, max(0, global_map.shape[0]-true_map.shape[0])], [0, max(0, global_map.shape[1]-true_map.shape[1])], [0, 0]])
                # global_map = np.pad(global_map, [[0, max(0, true_map.shape[0]-global_map.shape[0])], [0, max(0, true_map.shape[1]-global_map.shape[1])], [0, 0]], constant_values=0.5)
                # combined_map = 0.2 * true_map + 0.8 * global_map * np.array((1., 1., 0.))[None, None]  # yellow

                map_size = 300
                # Cut it to fixed size 300 x 300
                center_xy = (xy + target_xy) * 0.5
                desired_center_xy = np.array(map_size, np.float32) * 0.5
                center_xy = center_xy.astype(np.int)
                desired_center_xy = desired_center_xy.astype(np.int)

                offset_xy = (desired_center_xy - center_xy).astype(np.int)

                xy += offset_xy
                target_xy += offset_xy
                subgoal += offset_xy
                path += offset_xy[None]

                map_start_xy = np.maximum(center_xy - map_size//2, 0)
                map_cutoff_xy = -np.minimum(center_xy - map_size//2, 0)
                global_map = global_map[map_start_xy[0]:map_start_xy[0]+map_size-map_cutoff_xy[0], map_start_xy[1]:map_start_xy[1]+map_size-map_cutoff_xy[1]]

                combined_map = np.ones((map_size, map_size, 3), np.float32) * 0.5
                combined_map[map_cutoff_xy[0]:map_cutoff_xy[0] + global_map.shape[0], map_cutoff_xy[1]:global_map.shape[1]+map_cutoff_xy[1]] = global_map

                # global_map = global_map[:map_size-map_offset_xy[0], :map_size-map_offset_xy[1]]
                # combined_map[map_offset_xy[0]:map_offset_xy[0]+global_map.shape[0], map_offset_xy[1]:map_offset_xy[1]+global_map.shape[1]] = global_map

                combined_map[int(xy[0])-1:int(xy[0])+2, int(xy[1])-1:int(xy[1]+2)] = (0., 1., 0.)

                # print (self.video_ax.get_xlim())
                self.video_ax.set_xlim(-0.5, combined_map.shape[1]-0.5)
                self.video_ax.set_ylim(combined_map.shape[0]-0.5, -0.5)
                self.video_global_map_ax.set_data(combined_map)
                self.video_global_map_ax.set_extent([-0.5, combined_map.shape[1]-0.5, combined_map.shape[0]-0.5, -0.5])

                # # View angle
                # half_fov = 0.5 * np.deg2rad(70)
                # for ang_i, angle in enumerate([half_fov, -half_fov]):
                #     angle = angle - float(self.frame_traj_data[ind]['yaw'])
                #     # angle = angle + yaw[batch_i, traj_i, 0]
                #     v = np.array([np.cos(angle), np.sin(angle)]) * 10.
                #     x1 = np.array([xy[1], xy[0]])  # need to be flipped for display
                #     x2 = v + x1
                #
                #     self.video_view_angle_lines[ang_i].set_data([x1[0], x2[0]], [x1[1], x2[1]])
                #
                # # pdb.set_trace()
                # # Path
                # skip = 4
                # # # print(self.frame_traj_data[ind]['xy'], path[0])
                # # for i in range(len(self.video_path_circles)-2):
                # #     path_i = min(i * 4, len(path)-1)
                # #     xy = path[path_i]
                # #     self.video_path_circles[i].center = ([xy[1], xy[0]])
                # # # Sub-goal
                # # xy = self.frame_traj_data[ind]['subgoal']
                # # self.video_path_circles[-2].center = ([xy[1], xy[0]])
                # # # Target
                # # xy = path[-1]
                # # self.video_path_circles[-1].center = ([xy[1], xy[0]])
                # self.video_path_scatter.set_offsets(np.flip(path[::skip], axis=-1))
                # self.video_subgoal_scatter.set_offsets([np.flip(subgoal, axis=-1)])
                # self.video_target_scatter.set_offsets([np.flip(path[-1], axis=-1)])

            # self.video_text_ax2.set_data(self.summary_str)
        return self.video_image_ax

    def reset_video_writer(self):
        if len(self.frame_traj_data) > 0:
            # Save video
            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                self.video_image_ax = ax.imshow(np.zeros((90, 160, 3)))
                self.video_global_map_ax = None
                self.video_text_ax0 = fig.text(0.04, 0.9, self.summary_str, transform=fig.transFigure, fontsize=10, verticalalignment='top')  # bottom left

                self.video_text_ax1 = fig.text(0.96, 0.9, "Target", transform=fig.transFigure, fontsize=10, verticalalignment='top', horizontalalignment='right')
                self.video_text_ax2 = fig.text(0.04, 0.05, "Status2", transform=fig.transFigure, fontsize=10, verticalalignment='bottom', wrap=True)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(121)
                ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.video_image_ax = ax.imshow(np.zeros((90, 160, 3)))

                ax = fig.add_subplot(122)
                ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.video_global_map_ax = ax.imshow(np.zeros((1200, 1200, 3)))
                self.video_ax = ax

                # self.video_view_angle_lines = [mlines.Line2D([0., 0.], [10., 10.,], color='green') for _ in range(2)]
                # ax.add_line(self.video_view_angle_lines[0])
                # ax.add_line(self.video_view_angle_lines[1])
                # self.video_path_circles = []
                # for i in range(20):
                #     circle = plt.Circle((0., 0.), 2., color=('red' if i >= 18 else 'orange'), fill=False, transform='data')
                #     ax.add_artist(circle)
                #     self.video_path_circles.append(circle)
                # self.video_view_angle_lines = []
                # for _ in range(2):
                #     self.video_view_angle_lines.extend(ax.plot([0., 1.], [0., 1.], '-', color='blue'))  # plot returns a list of lines
                #
                # self.video_path_scatter = ax.scatter([0.], [1.], s=2., c='green', marker='o')
                # self.video_subgoal_scatter = ax.scatter([0.], [1.], s=2.5, c='green', marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'))
                # self.video_target_scatter = ax.scatter([0.], [1.], s=2., c='red', marker='o')

                self.video_text_ax0 = fig.text(0.04, 0.9, self.summary_str, transform=fig.transFigure, fontsize=10,
                                               verticalalignment='top')  # bottom left

                self.video_text_ax1 = fig.text(0.96, 0.9, "Target", transform=fig.transFigure, fontsize=10,
                                               verticalalignment='top', horizontalalignment='right')
                self.video_text_ax2 = fig.text(0.04, 0.05, "Status2", transform=fig.transFigure, fontsize=10,
                                               verticalalignment='bottom', wrap=True)



            # im.set_clim([0, 1])
            fig.set_size_inches([5, 5])
            plt.tight_layout()

            ani = animation.FuncAnimation(fig, self.video_update, len(self.frame_traj_data) * 3 + 21, interval=100)  # time between frames in ms. overwritted by fps below
            writer = animation.writers['ffmpeg'](fps=30)
            video_filename = os.path.join(self.logdir, '%s_rgb_%s_%d%s.mp4'%(self.scene_name, self.sim2real_track, self.episode_i, self.filename_addition))
            ani.save(video_filename, writer=writer, dpi=100)

            print ("Video saved to "+video_filename)
            self.num_videos_saved += 1

        self.frame_traj_data = []

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
        plt.imshow(local_map_pred * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0., vmax=1.)
        plt.axes(images_axarr[1, 1])
        if local_obj_map_pred is not None:
            plt.imshow(local_obj_map_pred * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0., vmax=1.)
        else:
            plt.imshow(local_map_label * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0., vmax=1.)
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
            self.yaw = np.array(yaw)

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
        xy = np.array(xy)
        yaw = np.array(yaw)

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
