import argparse

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
plt.ion()
import time
import os, sys

from agents.myagents import ExpertAgent, MappingAgent
import gibson2
import numpy as np
import json
from utils import logger

from arguments import parse_args

from multiprocessing import Process, Pool, Lock, Manager


SINGLE_PROCESS = True


def eval_helper(params, model_name, output_filename, logdir, write_lock=None, num_episodes_per_floor=4):
    from gibson2.envs.challenge import Challenge

    agent = MappingAgent(params, logdir=logdir, scene_name=str(model_name))
    challenge = Challenge()
    episode_infos = challenge.save_episodes(agent, output_filename=output_filename, models=[model_name],
                                            write_lock=write_lock, num_episodes_per_floor=num_episodes_per_floor,
                                            display_mode=params.gibson_display_mode)

    return episode_infos


def save_episodes_helper(model_name, output_filename, write_lock=None, num_episodes_per_floor=20):
    from gibson2.envs.challenge import Challenge

    agent = ExpertAgent()
    challenge = Challenge()
    challenge.save_episodes(agent, output_filename=output_filename, models=[model_name], write_lock=write_lock,
                            num_episodes_per_floor=num_episodes_per_floor)


def gen_map_helper(params, model_name):
    del params
    from gibson2.envs.challenge import Challenge

    agent = ExpertAgent()
    challenge = Challenge()
    episode_infos = challenge.generate_maps(agent, models=[model_name])

    return episode_infos


def wrapped_fun_with_return_dict(func, return_dict, *args, **kwargs):
    try:
        return_dict[0] = func(*args, **kwargs)
    except Exception as e:
        print("Exception")
        print (e.message)


def run_in_separate_process(func, *args, **kwargs):
    if SINGLE_PROCESS:
        return func(*args, **kwargs)
    else:
        manager = Manager()
        return_dict = manager.dict()

        p = Process(target=wrapped_fun_with_return_dict, args=(func, return_dict) + tuple(args), kwargs=kwargs)
        p.start()
        p.join()
        # TODO check for exceptions, now they are supressed
        p.terminate()

        try:
            return list(return_dict.values())[0]
        except IndexError:
            # import ipdb; ipdb.set_trace()
            return None


def main():
    params = parse_args(default_files=('./gibson_submission.conf', ))
    is_submission = (params.gibson_mode == 'submission')

    if is_submission:
        from gibson2.envs.challenge import Challenge

        challenge = Challenge()
        agent = MappingAgent(params)
        challenge.submit(agent)

        print ("Done with submission.")
        raise SystemExit

    if params.gibson_track != '':
        os.environ['SIM2REAL_TRACK'] = params.gibson_track
    track = os.environ['SIM2REAL_TRACK']

    # challenge.submit(agent)
    models = sorted(os.listdir(gibson2.dataset_path))
    skip = []
    if params.gibson_split == "custom":
        # # models = models[:5]
        # models = models[14:15]
        # # models = models[-10:]
        # num_episodes_per_floor = 1
        # num_repeats = 1
        models = models[:-10]
        num_episodes_per_floor = 5
        num_repeats = 8
        skip = range(708)
    elif params.gibson_split == "train":
        # models = models[67:-10] + models[:66]
        # models = models[:66] + models[67:-10]
        models = models[:-10]
        num_episodes_per_floor = 5
        num_repeats = 4
    elif params.gibson_split == "test":
        models = models[-10:]
        num_episodes_per_floor = 5
        num_repeats = 4
        # models = models[-3:]
    elif params.gibson_split == "evaltest":
        models = models[-10:]
        num_episodes_per_floor = 10
        num_repeats = 1
        # models = models[-3:]
    elif params.gibson_split == "minitest":
        models = models[-10:]
        num_episodes_per_floor = 5
        num_repeats = 1
        # models = models[-3:]
    else:
        raise ValueError("Unknown split %s"%params.gibson_split)

    models = models

    if params.gibson_mode == 'gen_maps':
        num_repeats = 1

        print (models)
        assert len(models) < 1000

        print ("Generating maps for %d models."%len(models))
        for model_i, model_name in enumerate(models):
            # p = Process(target=save_episodes_helper, args=(args, model_name, output_filename + '.{:03d}'.format(i), num_episodes_per_floor=num_episodes_per_floor))
            # p.start()
            # p.join()
            # p.terminate()
            if model_i in skip:
                continue
            run_in_separate_process(gen_map_helper, params, model_name)

    elif params.gibson_mode in ['gen_scenarios']:
        models = models * num_repeats
        print (models)
        assert len(models) < 1000

        output_filename = './data/scenarios_{}_{}_{}.tfrecords'.format(params.gibson_split, track, time.strftime('%m-%d-%H-%M-%S', time.localtime()))

        for model_i, model_name in enumerate(models):
            if model_i in skip:
                continue
            run_in_separate_process(save_episodes_helper, model_name, output_filename + '.{:03d}'.format(model_i), num_episodes_per_floor=num_episodes_per_floor)

    elif params.gibson_mode in ['eval']:
        models = models * num_repeats
        print (models)
        assert len(models) < 1000

        evaluate_episodes(params, models, num_episodes_per_floor)

    elif params.gibson_mode in ['evalsubmission']:
        models = models * num_repeats
        print (models)
        assert len(models) < 1000
        from gibson2.envs.challenge import Challenge

        for model_i, model_name in enumerate(models):
            challenge = Challenge()
            agent = MappingAgent(params)
            challenge.submit(agent)

    else:

        raise ValueError("Unknown gibson_mode=%s"%params.gibson_mode)


def evaluate_episodes(params, models, num_episodes_per_floor):
    track = os.environ['SIM2REAL_TRACK']
    timestamp_str = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    logdir = './temp/evals/{}_{}'.format(track, timestamp_str)
    os.makedirs(logdir)

    output_filename = os.path.join(logdir, 'data.tfrecords')
    summary_filename = os.path.join(logdir, 'stat')

    sys.stdout = logger.Logger(os.path.join(logdir, "out.log"))

    episode_infos = []
    for model_i, model_name in enumerate(models):
        ep_info = run_in_separate_process(eval_helper, params, model_name, output_filename + '.{:03d}'.format(model_i),
                                          logdir=logdir,
                                          num_episodes_per_floor=num_episodes_per_floor)
        if ep_info is None:
            print ("There was an error in evaluation for model %s. Skipping."%model_name)
            continue

        episode_infos.extend(list(ep_info))

        # This is to limit memory leak
        # p = Process(target=save_episodes_helper, args=(args, model_name, output_filename + '.{:03d}'.format(i), num_episodes_per_floor=num_episodes_per_floor))
        # p.start()
        # p.join()
        # p.terminate()

        # Print summary after every model. This way partial results are saved even if there is an error.
        episode_infos_dict = {key: [] for key in episode_infos[0].keys()}
        for episode_info in episode_infos:
            for key, val in episode_info.items():
                episode_infos_dict[key].append(val)

        episode_infos_dict['has_collided'] = [(1. if val > 0 else 0.) for val in episode_infos_dict['collision_step']]
        # collision before moving less then 10 cm (2x5cm) from initial pose
        episode_infos_dict['has_collided_but_not_moved'] = [
            (1. if episode_infos_dict['collision_step'][i] > 0 and episode_infos_dict['distance_from_start'][i] <= 2. else 0.)
            for i in range(len(episode_infos_dict['collision_step']))]
        episode_infos_dict['success_strict'] = [(1. if succ > 0 and col < 1. else 0.) for succ, col in
                                                zip(episode_infos_dict['success'], episode_infos_dict['has_collided'])]
        summary_dict = {}
        summary_dict['num_episodes'] = len(episode_infos_dict['success'])
        summary_dict['num_floors'] = len(set(zip(episode_infos_dict['model_id'], episode_infos_dict['floor'])))
        summary_dict['num_models'] = len(set(episode_infos_dict['model_id']))
        summary_dict['track'] = track

        keys_to_mean = ['success', 'success_strict', 'has_collided', 'episode_length', 'spl', 'is_colliding', 'has_collided_but_not_moved', 'is_colliding_but_trav_map_free', 'is_colliding_but_scan_map_free', 'cpu_time', 'path_length', 'timeout', 'collisions_allowed']
        for key in keys_to_mean:
            summary_dict['mean_'+key] = np.mean(episode_infos_dict[key])

        summary_dict['max_cpu_time_overall'] = np.max(episode_infos_dict['max_cpu_time'])

        episode_infos_dict.update(summary_dict)

        episode_infos_dict['params'] = str(params)

        print ("%d/%d: Success %f; Stict: %f; Col %f; EpisodeLen: %f; SPL %f" % (
            episode_infos_dict['num_models'], len(models),
            episode_infos_dict['mean_success'], episode_infos_dict['mean_success_strict'],
            episode_infos_dict['mean_has_collided'], episode_infos_dict['mean_episode_length'],
            episode_infos_dict['mean_spl'],))

        filename = summary_filename + '.detailed.json'
        with open(filename, 'w') as file:
            json.dump(episode_infos_dict, file, indent=4)
        filename = summary_filename + '.summary.json'
        with open(filename, 'w') as file:
            json.dump(summary_dict, file, indent=4)
        print (filename)
        print (summary_dict)


if __name__ == "__main__":
    try:
        main()
    finally:
        if isinstance(sys.stdout, logger.Logger):
            sys.stdout.closefile()
            sys.stdout = sys.stdout.terminal
