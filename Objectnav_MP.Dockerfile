FROM fairembodied/habitat:latest

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    htop \
    tmux \
    unzip &&\
    rm -rf /var/lib/apt/lists/*


RUN /bin/bash -c ". activate habitat; pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install numpy"

RUN /bin/bash -c ". activate habitat; pip install matplotlib  seaborn==0.9.0 scikit-fmm==2019.1.30 scikit-image imageio==2.6.0 scikit-learn==0.22.2.post1 ifcfg"

RUN /bin/bash -c "apt-get update; apt-get install -y libsm6 libxext6 libxrender-dev; pip install opencv-python"

RUN /bin/bash -c ". activate habitat; conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge"

# RUN /bin/bash -c ". activate habitat; python -m pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html"

#python main.py  --split val --eval 1  --num_processes_per_gpu 1  --num_processes_on_first_gpu 2  --num_eval_episodes 50    --load  /DATA/disk1/epic/jiazhaozhang/navigation_data/ObjectGoalNav_data/logs/dump/exp1/periodic_500004.pth

ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"




ADD RL_agent.py agent.py
ADD constants.py constants.py
ADD arguments.py arguments.py



ADD code_backup/algo  /algo
ADD code_backup/envs  /envs
ADD code_backup/utils  /utils
ADD code_backup/configs /configs
ADD code_backup/agents /agents
ADD code_backup/legend.png legend.png






ADD code_backup/model.py model.py


ADD model_best.pth model_best.pth
# ADD model_final_c10459.pkl model_final_c10459.pkl
ADD rednet_semmap_mp3d_tuned.pth rednet_semmap_mp3d_tuned.pth

ADD submission.sh submission.sh
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml


ENV AGENT_EVALUATION_TYPE remote

# ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && bash submission.sh"]
