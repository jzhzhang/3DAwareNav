# 3D-Aware Object Goal Navigation via Simultaneous Exploration and Identification 


We propose a framework for the challenging 3D-aware ObjectNav based on two straightforward sub-policies. The two sub-polices, namely corner-guided exploration policy and category-aware identification policy, simultaneously perform by utilizing online fused 3D points as observation.



## Setup
- *Dependeces*: We use earlier (`0.2.2`) versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.2) and [habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.2). Other related depencese can be found in `requirements.txt`. 

- *Data (MatterPort3D)*: Please download the scene dataset and the episode dataset from [habitat-lab/DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset). Then organize the files as follows:
```
3dAwareNav/
  data/
    scene_datasets/
        mp3d/
    episode_datasets/
        objectnav_mp3d_v1/
```
The weight of our 2D backbone RedNet can be found in [Stubborn](https://github.com/Improbable-AI/Stubborn).


## Training and Evaluating:

We provide scripts for quick training and evaluation. The parameters can be found in [sh_train_mp3d.sh](sh_train_mp3d.sh) and [sh_eval.sh](sh_eval.sh), You can modify these parameters to customize them according to your specific requirements.
```
sh sh_train_mp3d.sh # training 
sh sh_eval.sh # evaluating
```

## Citation
```
@inproceedings{zhang20233d,
  title={3D-Aware Object Goal Navigation via Simultaneous Exploration and Identification},
  author={Zhang, Jiazhao and Dai, Liu and Meng, Fanpeng and Fan, Qingnan and Chen, Xuelin and Xu, Kai and Wang, He},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6672--6682},
  year={2023}
}
```

## Acknowledgments

Our code is inspired by [Object-Goal-Navigation](https://github.com/devendrachaplot/Object-Goal-Navigation) and [FusionAwareConv](https://github.com/jzhzhang/FusionAwareConv) .

This is an open-source version, some functions have been rewritten to avoid certain license. It would not be expected to reproduce the result exactly, but the result is almost the same. Thank Liu Dai (@bbbbbMatrix) and Fanpeng Meng (@mfp0610) for their contributions to this repository.


## Contact

If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.





