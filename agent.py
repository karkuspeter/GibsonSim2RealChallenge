import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from agents.myagents import MyAgent, get_agent

from gibson2.envs.challenge import Challenge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="Random", choices=["Random", "ForwardOnly", "SAC"])
    parser.add_argument("--ckpt-path", default="", type=str)
    args = parser.parse_args()

    agent = get_agent(
        agent_class=args.agent_class,
        ckpt_path=args.ckpt_path
    )
    challenge = Challenge()
    # challenge.submit(agent)
    challenge.save_episodes(agent)


if __name__ == "__main__":
    main()
