#!/usr/bin/env bash

DOCKER_NAME="my_submission_clean2"
GIBSON_DATA_PATH="../GibsonSim2RealChallengeClean/gibson-challenge-data"
CONFIG_EXTENSION=""
SIM2REAL_TRACK="static"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
      --data-path)
      shift
      GIBSON_DATA_PATH="${1}"
      CONFIG_EXTENSION=".challenge"
      shift
      ;;
      --track)
      shift
      SIM2REAL_TRACK="${1}"
      shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

    # -v $(pwd)/../GibsonEnvV2:/opt/GibsonEnvV2 \
    # -v $(pwd)/agents:/agents \

docker run -ti -v "$(pwd)/$GIBSON_DATA_PATH":/gibson-challenge-data \
    -v $(pwd)/temp:/temp \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml$CONFIG_EXTENSION; export SIM2REAL_TRACK=$SIM2REAL_TRACK; cp /gibson-challenge-data/global_config.yaml$CONFIG_EXTENSION /opt/GibsonEnvV2/gibson2/global_config.yaml; ipython agent.py --pdb -- --gibson_mode evalsubmission --gibson_split evaltest; bash"

