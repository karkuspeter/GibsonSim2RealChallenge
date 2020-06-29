import numpy as np
from scipy.optimize import least_squares

import argparse
import re
import pickle
from utils.dotdict import dotdict
import matplotlib.pyplot as plt
plt.ion()

try:
    import ipbd as pdb
except:
    import pdb


def parse_logfile(filenames, source='real', model='lin1'):

    trajectories = []
    states = None
    actions = None

    linear_velocity_scaler = 0.5
    angular_velocity_scaler = 1.5707963267948966

    if isinstance(filenames, str):
        filenames = [filenames]

    t = 0
    re_spin = re.compile(r"^([0-9]+): spin.*: ([0-9\-\.]+)")
    re_action = re.compile(r"^([0-9]+): .* POS ([0-9\-\.]+) ([0-9\-\.]+) --.*\] ([0-9\-\.]+) besti")
    re_sys = re.compile(r"\[real\]\[info\] action step: ([0-9]+) \| output: \[ *([0-9\-\.]+) +([0-9\-\.]+) *\]")
    re_sym_action1 = re.compile(
        r"^([0-9]+): .* POS ([0-9\-\.]+) ([0-9\-\.]+) --.*\] ([0-9\-\.]+) besti.* \| \[([0-9\-\.]+) ([0-9\-\.]+)\]")
    re_sym_action2 = re.compile(
        r"^([0-9]+): delay[0-9]+ ([0-9\-\.]+) ([0-9\-\.]+) ([0-9\-\.]+) act.* act ([0-9\-\.]+) ([0-9\-\.]+)")  # extract the real action
    re_sym_action3 = re.compile(
        r"^([0-9]+): delay[0-9]+ ([0-9\-\.]+) ([0-9\-\.]+) ([0-9\-\.]+) act ([0-9\-\.]+) ([0-9\-\.]+)")  # extract the intended action

    for filename in filenames:
        with open(filename, 'r') as file:

            for line_i in range(100000):
                line = file.readline()

                # Reset
                m_reset = re.match(r"^Resetting agent", line)
                if m_reset:
                    if states is not None:
                        trajectories.append((states[:t+1], actions[:t+1]))
                    states = np.ones((500, 3), np.float) * np.nan
                    actions = np.ones((500, 2), np.float) * np.nan
                    t = 0

                # Steps
                if source == 'sim':
                    m_sym_action = re_sym_action3.match(line)  # intended action

                    if m_sym_action is not None:
                        t, x, y, yaw, act_fwd, act_rot = m_sym_action.groups()
                        t = int(t)
                        assert np.all(np.isnan(states[t]))
                        states[t] = [float(x), float(y), np.deg2rad(float(yaw))]
                        assert np.all(np.isnan(actions[t]))
                        actions[t] = (float(act_fwd), float(act_rot))
                        # print(line, states[t])

                else:
                    m_spin = re_spin.match(line)
                    m_action = re_action.match(line)
                    m_sys = re_sys.match(line)
                    m_sym_action = re_sym_action1.match(line)

                    if m_spin is not None:
                        t, yaw = m_spin.groups()
                        t = int(t)
                        assert np.all(np.isnan(states[t]))
                        states[t, 2] = float(yaw)
                        print(line, states[t])

                    if m_sym_action is not None:
                        t, x, y, yaw, act_fwd, act_rot = m_sym_action.groups()
                        t = int(t)
                        assert np.all(np.isnan(states[t]))
                        states[t] = [float(x), float(y), np.deg2rad(float(yaw))]
                        assert np.all(np.isnan(actions[t]))
                        actions[t] = (float(act_fwd), float(act_rot))
                        # print(line, states[t])
                    else:
                        if m_action is not None:
                            t, x, y, yaw = m_action.groups()
                            t = int(t)
                            assert np.all(np.isnan(states[t]))
                            states[t] = [float(x), float(y), np.deg2rad(float(yaw))]
                            print(line, states[t])

                        if m_sys is not None:
                            assert m_spin is None and m_action is None
                            t, act_fwd, act_rot = m_sys.groups()
                            t = int(t)-1
                            assert np.all(np.isnan(actions[t]))
                            actions[t] = (float(act_fwd), float(act_rot))
                            print(line, actions[t])

    print ("done")

    clean_trajectories = []
    for states, actions in trajectories:
        if np.all(np.isnan(states)):
            continue

        if len(states) < 2:
            continue

        # traj = np.concatenate([states, actions], axis=-1)
        lin_vel = np.linalg.norm(states[1:, :2] - states[:-1, :2], axis=-1)
        ang_vel = states[1:, 2] - states[:-1, 2]
        ang_vel = (ang_vel + np.pi) % (2*np.pi) - np.pi
        act_fwd = actions[:, 0]  # * linear_velocity_scaler * 0.1
        act_rot = actions[:, 1]  # * angular_velocity_scaler * 0.1

        traj = dotdict(dict(
            x=states[:, 0], y=states[:, 1], yaw=states[:, 2], lin_vel=lin_vel, ang_vel=ang_vel,
            act_fwd=act_fwd, act_rot=act_rot, trajlen=len(states),
        ))
        clean_trajectories.append(traj)
        # print (traj)

    # time_delay = 2
    # action_fwd_rescaler = 0.3
    # action_rot_rescaler = 0.5

    errors = []
    scalers = []

    for time_delay in range(5):
        lin_vel_list = []
        act_fwd_list = []
        ang_vel_list = []
        act_rot_list = []
        for traj in clean_trajectories:
            valid_part_start = np.min(np.flatnonzero(np.isfinite(traj.lin_vel)))
            # act_t is the reference velocity at t+delay. For one step prediction, we want pairs act[i], vel[i+delay]
            lin_vel = traj.lin_vel[valid_part_start + time_delay:]
            act_fwd = traj.act_fwd[valid_part_start:traj.act_fwd.shape[0] - time_delay - 1]
            ang_vel = traj.ang_vel[valid_part_start + time_delay:]
            act_rot = traj.act_rot[valid_part_start:traj.act_rot.shape[0] - time_delay - 1]

            assert len(lin_vel) == len(act_fwd)
            assert len(ang_vel) == len(act_rot)
            assert len(ang_vel) == len(lin_vel)

            lin_vel_list.append(lin_vel)
            act_fwd_list.append(act_fwd)
            ang_vel_list.append(ang_vel)
            act_rot_list.append(act_rot)

        lin_vel = np.concatenate(lin_vel_list)
        act_fwd = np.concatenate(act_fwd_list)
        ang_vel = np.concatenate(ang_vel_list)
        act_rot = np.concatenate(act_rot_list)

        pred_func, sc, err = sysid(lin_vel, ang_vel, act_fwd, act_rot, model=model)
        scalers.append(sc)
        errors.append(err)

    ang_error_scaler = np.mean(np.abs(lin_vel)) / np.mean(np.abs(ang_vel))
    errors = errors * np.array([1., np.square(ang_error_scaler)])[None]
    time_delay = np.argmin(np.sum(errors, axis=-1))
    print (errors)
    print (time_delay, scalers[time_delay], errors[time_delay])
    # (3, (0.6540917264778654, 0.07866921965909571), array([42.36897857, 36.7125886]))

    best_scalers = scalers[time_delay]  # 0.6540917264778654, 0.07866921965909571
    fwd_scaler, rot_scaler =  0.6540917264778654, 0.07866921965909571

    # plots one-step predictions with time-delay
    plt.close('all')
    for traj in clean_trajectories[:10]:
        # act_t is the reference velocity at t+delay. we want pairs act[i], vel[i+delay]
        act_fwd = np.pad(traj.act_fwd, [[time_delay, 0]], 'constant')  # add zeros to beginning
        act_rot = np.pad(traj.act_rot, [[time_delay, 0]], 'constant')
        act_fwd = act_fwd[:-time_delay-1]  # drop last actions, we have not seen their effect since the episode terminated
        act_rot = act_rot[:-time_delay-1]

        # lin_vel_tmo[t] = vel[t-1]
        lin_vel_tmo = np.pad(traj.lin_vel[:-1], [[1, 0], ], 'constant')
        ang_vel_tmo = np.pad(traj.ang_vel[:-1], [[1, 0], ], 'constant')

        pred_xy, pred_yaw = pred_func(best_scalers, lin_vel_tmo, act_fwd, ang_vel_tmo, act_rot)

        plt.figure()
        plt.plot(np.arange(len(traj.lin_vel)), np.zeros_like(traj.lin_vel), color='black', marker='', linestyle='-', linewidth=1)
        plt.plot(np.arange(len(traj.lin_vel)), traj.lin_vel, color='blue', marker='.', linestyle='-', linewidth=1.2)
        plt.plot(np.arange(len(pred_xy)), pred_xy, color='green', marker='.', linestyle='-', linewidth=1.2)
        plt.plot(np.arange(len(act_fwd)), act_fwd * fwd_scaler, color='red', marker='.', linestyle='-', linewidth=1.2)
        plt.ylim([-0.1, 1.5])

        plt.figure()
        plt.plot(np.arange(len(traj.ang_vel)), np.zeros_like(traj.ang_vel), color='black', marker='', linestyle='-', linewidth=1)
        plt.plot(np.arange(len(traj.ang_vel)), np.rad2deg(traj.ang_vel), color='blue', marker='.', linestyle='-', linewidth=1.2)
        plt.plot(np.arange(len(pred_yaw)), np.rad2deg(pred_yaw), color='green', marker='.', linestyle='-', linewidth=1.2)
        plt.plot(np.arange(len(act_rot)), np.rad2deg(act_rot * rot_scaler), color='red', marker='.', linestyle='-', linewidth=1.2)
        plt.ylim([-10, 10])

    plt.show()
    pdb.set_trace()

    # for m in [m_spin, m_action, m_sys]:
    #     if m is not None:
    #         print (str(m.groups()) + " | " + line)


def lin1_model(x, v_tmo, reference_vel):
    del v_tmo
    return x[0] * reference_vel


def lin2_model(x, v_tmo, reference_v_t):
    # v_t = x0 * v_{t-1} + x1 * u_t
    pred = x[0] * v_tmo + x[1] * reference_v_t
    return pred


def error_func(x, v_t, pred_func, *args):
    pred = pred_func(x, *args)
    return pred - v_t


def sysid(lin_vel, ang_vel, act_fwd, act_rot, model):
    lin_vel_tmo = np.pad(lin_vel[:-1], [[1, 0], ], 'constant')
    ang_vel_tmo = np.pad(ang_vel[:-1], [[1, 0], ], 'constant')

    if model == 'lin1':
        pred_func = lin1_model
        res_lin = least_squares(error_func, np.array((5.,)), args=(lin_vel, pred_func, lin_vel_tmo, act_fwd), loss='linear')
        res_rot = least_squares(error_func, np.array((0.5,)), args=(ang_vel, pred_func, ang_vel_tmo, act_rot), loss='linear')

    elif model == 'lin2':
        pred_func = lin2_model
        res_lin = least_squares(error_func, np.array((1.,  2.,)), args=(lin_vel, pred_func, lin_vel_tmo, act_fwd), loss='linear')
        res_rot = least_squares(error_func, np.array((1., 0.5,)), args=(ang_vel, pred_func, ang_vel_tmo, act_rot), loss='linear')

    else:
        raise ValueError(model)

    joint_pred_func = lambda x, linvel_tmo, actfwd, angvel_tmo, actrot: (pred_func(x[:len(x)//2], linvel_tmo, actfwd), pred_func(x[len(x)//2:], angvel_tmo, actrot))

    scalers = tuple(list(res_lin.x) + list(res_rot.x))
    errors = (res_lin.cost, res_rot.cost)
    return joint_pred_func, scalers, errors


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('filename', help="log file")

    args = p.parse_args()

    with open(args.filename, 'rb') as file:
        x = pickle.load(file)
        pdb.set_trace()

    parse_logfile(args.filename)
