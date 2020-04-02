
from agents.simple_agent import RandomAgent, ForwardOnlyAgent
from agents.rl_agent import SACAgent

import matplotlib.pyplot as plt
import time


class MyAgent(RandomAgent):
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


def get_agent(agent_class, ckpt_path=""):
    return MyAgent()
    if agent_class == "Random":
        return RandomAgent()
    elif agent_class == "ForwardOnly":
        return ForwardOnlyAgent()
    elif agent_class == "SAC":
        return SACAgent(root_dir=ckpt_path)
