#!/usr/bin/env bash


# export CONFIG_FILE=/home/peter/code/GibsonSim2RealChallenge/gibson-challenge-data/locobot_p2p_nav_house.yaml; export SIM2REAL_TRACK=static; export PYTHONPATH=/mclnet:$PYTHONPATH;
# ipython agent.py --pdb -- -c ./mclnet/mapper4.conf --brain mapperbrain_v16  --load ./data/gibson/mapperbrain_v16-xxx-both-0-0-odom1-2-map-500-1-1-lr001-b64-def-fixcoord2-0 --testfile mapping-v4-20-sh/mapping-test-v2.tfrecords  --trajlen 1 --batchsize 1 --mode both

ipython agent.py --pdb

