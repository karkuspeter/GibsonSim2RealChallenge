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

from arguments import parse_args

from multiprocessing import Process, Pool, Lock


def eval_helper(params, model_name, output_filename, write_lock=None):
    from gibson2.envs.challenge import Challenge

    agent = MappingAgent(params)
    challenge = Challenge()
    challenge.save_episodes(agent, output_filename=output_filename, models=[model_name], write_lock=write_lock)


def main():
    params = parse_args()
    split = "train"

    # challenge.submit(agent)
    output_filename = './data/eval_{}.tfrecords'.format(time.strftime('%m-%d-%H-%M-%S', time.localtime()))
    models = sorted(os.listdir(gibson2.dataset_path))
    if split == "train":
        # models = models[67:-10] + models[:66]
        models = models[:66] + models[67:-10]
    else:
        models = models[-10:]
    models = models * 5
    print (models)
    assert len(models) < 1000

    for i, model_name in enumerate(models):
        eval_helper(params, model_name, output_filename)


        # This is to limit memory leak
        # p = Process(target=save_episodes_helper, args=(args, model_name, output_filename + '.{:03d}'.format(i)))
        # p.start()
        # p.join()
        # p.terminate()


if __name__ == "__main__":
    main()
