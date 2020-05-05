FROM gibsonchallenge/gibsonv2:latest
ENV PATH /miniconda/envs/gibson/bin:$PATH
ENV PYTHONPATH /mclnet$PYTHONPATH

RUN pip install ipdb zmq flask
RUN pip install tensorflow-gpu==1.15 tensorpack configargparse socketIO-client
RUN mkdir /temp

# COPY ../GibsonSim2RealChallenge/agent.py /agent.py
# ADD simple_agent.py /simple_agent.py
# ADD rl_agent.py /rl_agent.py

# COPY submission.sh /submission.sh

#COPY ../GibsonSim2RealChallenge/__init__.py /__init__.py
COPY *.py /
COPY *.sh /
COPY *.conf /
COPY agents /agents
COPY data/gibson /data/gibson
COPY build/mclnet /mclnet


WORKDIR /
