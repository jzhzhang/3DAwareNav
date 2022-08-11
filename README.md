# Navigation with novel 3D scene representation [On-going]



## Installing Dependencies
- We use earlier versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below:

Installing habitat-sim:
```
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)
```

Installing habitat-lab:
```
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; 
pip install -e .
```
Check habitat installation by running `python examples/benchmark.py` in the habitat-lab folder.

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on pytorch v1.10.1 and cudatoolkit v11.1. If you are using conda:
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration. If you are using conda:
```
python -m pip install detectron2==0.6 -f \\n  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

### Docker and Singularity images:
We provide experimental [docker](https://www.docker.com/) and [singularity](https://sylabs.io/) images with all the dependencies installed, see [Docker Instructions](./docs/DOCKER_INSTRUCTIONS.md).


## Setup
Download the code and install other requirements:
```
pip install -r requirements.txt
```

### Downloading exisitng dataset (all the data are well-organized)
```
/DATA/disk1/epic/jiazhaozhang/habitat-lab_data/ObjectGoalNav/data
```

Currentlt, we only consider following sequences:



### Training:
```
sh run_script_server.sh
```

### Debugging Testing:
```
sh test_script_server.sh
```





