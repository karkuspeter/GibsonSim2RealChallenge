#!/usr/bin/env bash

DOCKER_NAME="my_submission"
X11=0

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    --x11)
      shift
      X11=1
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

if [ $X11 == 1 ]
then
./x11docker/x11docker -i --runtime=nvidia -- \
    -v $(pwd)/gibson-challenge-data:/gibson-challenge-data \
    -v $(pwd)/agents:/agents \
    -v $(pwd)/GibsonEnvV2:/opt/GibsonEnvV2 \
    -p 5001:5001 \
    -it \
    -- ${DOCKER_NAME} \
    /bin/bash -c "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml; export SIM2REAL_TRACK=static; cp /gibson-challenge-data/global_config.yaml /opt/GibsonEnvV2/gibson2/global_config.yaml; bash"
else
 docker run -v $(pwd)/gibson-challenge-data:/gibson-challenge-data \
    -v $(pwd)/agents:/agents \
    -v $(pwd)/GibsonEnvV2:/opt/GibsonEnvV2 \
    -p 5001:5001 \
    -it \
    ${DOCKER_NAME} \
    /bin/bash -c "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml; export SIM2REAL_TRACK=static; cp /gibson-challenge-data/global_config.yaml /opt/GibsonEnvV2/gibson2/global_config.yaml; bash"
fi
    #--runtime=nvidia \

#    -v $(pwd)/GibsonEnvV2:/opt/GibsonEnvV2 \
