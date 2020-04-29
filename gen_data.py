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

from agents.myagents import ExpertAgent, get_agent
import gibson2

from multiprocessing import Process, Pool, Lock


def save_episodes_helper(model_name, output_filename, write_lock=None):
    from gibson2.envs.challenge import Challenge

    agent = ExpertAgent()
    challenge = Challenge()
    challenge.save_episodes(agent, output_filename=output_filename, models=[model_name], write_lock=write_lock)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--agent-class", type=str, default="Random", choices=["Random", "ForwardOnly", "SAC"])
    # parser.add_argument("--ckpt-path", default="", type=str)
    parser.add_argument("--split", default="train", type=str)
    args = parser.parse_args()

    # challenge.submit(agent)
    output_filename = './data/scenarios_{}.tfrecords'.format(time.strftime('%m-%d-%H-%M-%S', time.localtime()))
    models = sorted(os.listdir(gibson2.dataset_path))
    if args.split == "train":
        models = models[67:-10] + models[:66]
    else:
        models = models[-10:]
    models = models * 5
    print (models)
    assert len(models) < 1000

    # write_lock = Lock()
    # pool = Pool(processes=1, maxtasksperchild=1)  #  pybullet doesn't work with more than 1 process

    # for model_name in models:
    #     # save_episodes_helper(args, model_name, output_filename, write_lock)
    #     pool.apply_async(save_episodes_helper, args=(model_name, output_filename, write_lock))

    # time.sleep(5)
    # pool.close()
    # pool.join()

    for i, model_name in enumerate(models):
        # This is to limit memory leak
        p = Process(target=save_episodes_helper, args=(model_name, output_filename + '.{:03d}'.format(i)))
        p.start()
        p.join()
        p.terminate()


if __name__ == "__main__":
    main()
