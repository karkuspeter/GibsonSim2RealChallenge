import argparse

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
plt.ion()
import time
import os

from agents.myagents import ExpertAgent, MappingAgent
import gibson2
import numpy as np
import json

from arguments import parse_args

from multiprocessing import Process, Pool, Lock
from gibson2.envs.challenge import Challenge


def eval_helper(params, model_name, output_filename, write_lock=None, num_episodes_per_floor=4):

    agent = MappingAgent(params)
    challenge = Challenge()
    episode_infos = challenge.save_episodes(agent, output_filename=output_filename, models=[model_name],
                                            write_lock=write_lock, num_episodes_per_floor=num_episodes_per_floor)

    return episode_infos


def gen_map_helper(params, model_name):
    del params

    agent = ExpertAgent()
    challenge = Challenge()
    episode_infos = challenge.generate_maps(agent, models=[model_name])

    return episode_infos


def main():
    params = parse_args(default_files=('./gibson_submission.conf', ))
    is_submission = (params.gibson_mode == 'submission')

    if is_submission:
        challenge = Challenge()
        agent = MappingAgent(params)
        challenge.submit(agent)

        print ("Done with submission.")
        raise SystemExit

    # challenge.submit(agent)
    models = sorted(os.listdir(gibson2.dataset_path))
    if params.gibson_split == "custom":
        # models = models[:5]
        models = models[-10:]
        num_episodes_per_floor = 1
    elif params.gibson_split == "train":
        # models = models[67:-10] + models[:66]
        # models = models[:66] + models[67:-10]
        models = models[:-10]
        num_episodes_per_floor = 200
    else:
        models = models[-10:]
        num_episodes_per_floor = 10
        # models = models[-3:]
    models = models * 1

    print (models)
    assert len(models) < 1000

    if params.gibson_mode == 'gen_maps':
        print ("Generating maps for %d models."%len(models))
        for model_i, model_name in enumerate(models):
            # p = Process(target=save_episodes_helper, args=(args, model_name, output_filename + '.{:03d}'.format(i), num_episodes_per_floor=num_episodes_per_floor))
            # p.start()
            # p.join()
            # p.terminate()

            gen_map_helper(params, model_name)
    elif params.gibson_mode in ['submission', 'eval']:
        evaluate_episodes(params, models, num_episodes_per_floor)
    else:
        raise ValueError("Unknown gibson_mode=%s"%params.gibson_mode)


def evaluate_episodes(params, models, num_episodes_per_floor):
    output_filename = './data/eval_{}.tfrecords'.format(time.strftime('%m-%d-%H-%M-%S', time.localtime()))

    episode_infos = []
    for model_i, model_name in enumerate(models):
        episode_infos.extend(list(
            eval_helper(params, model_name, output_filename + '.{:03d}'.format(model_i),
                        num_episodes_per_floor=num_episodes_per_floor)))

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
        episode_infos_dict['success_strict'] = [(1. if succ > 0 and col < 1. else 0.) for succ, col in
                                                zip(episode_infos_dict['success'], episode_infos_dict['has_collided'])]

        episode_infos_dict['num_episodes'] = len(episode_infos_dict['success'])
        episode_infos_dict['num_floors'] = len(set(zip(episode_infos_dict['model_id'], episode_infos_dict['floor'])))
        episode_infos_dict['num_models'] = len(set(episode_infos_dict['model_id']))
        episode_infos_dict['mean_success'] = np.mean(episode_infos_dict['success'])
        episode_infos_dict['mean_success_strict'] = np.mean(episode_infos_dict['success_strict'])
        episode_infos_dict['mean_collided'] = np.mean(episode_infos_dict['has_collided'])
        episode_infos_dict['mean_episode_length'] = np.mean(episode_infos_dict['episode_length'])
        episode_infos_dict['mean_spl'] = np.mean(episode_infos_dict['spl'])

        print ("%d/%d: Success %f; Stict: %f; Col %f; EpisodeLen: %f; SPL %f" % (
            episode_infos_dict['num_models'], len(models),
            episode_infos_dict['mean_success'], episode_infos_dict['mean_success_strict'],
            episode_infos_dict['mean_collided'], episode_infos_dict['mean_episode_length'],
            episode_infos_dict['mean_spl'],))

        filename = output_filename.replace('.tfrecords', '.summary.json')
        if filename == output_filename:
            filename += '.summary.json'
        with open(filename, 'w') as file:
            json.dump(episode_infos_dict, file, indent=4)
        print (filename)


if __name__ == "__main__":
    main()
