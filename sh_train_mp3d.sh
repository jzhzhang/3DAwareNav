
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

# <<<<<<< HEAD
# # python main.py --auto_gpu_config 0 -n 8 --sem_gpu_id "cuda:3"  --policy_gpu_id "cuda:0"  --sim_gpu_id "0,0,2,3"  --split train --backbone_2d  "rednet" --stop_policy "3D" --print_images 1  -d ./tmp --exp_name exp_kl_goal  --save_periodic 20000 

# python main.py --auto_gpu_config 0 -n 8 --sem_gpu_id_list "2"  --policy_gpu_id "cuda:3" \
#     --sim_gpu_id "2"  --split train --backbone_2d  "rednet" --stop_policy "3D" \
#     --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml" --dataset "mp3d" \
#     --num_sem_categories 22 --deactivate_entropymap\
#     --print_images 1  -d ./new_tmp --exp_name exp_3d_policy_mp  --save_periodic 20000 \
#     --load_2d "/home/jiazhaozhang/project/navigation/habitat-challenge/periodic_590000.pth" \
#     --load_3d "/DATA/disk1/epic/jiazhaozhang/navigation_data/new_tmp/dump/exp_3d_policy_mp-09-09-23:22:07/periodic_3d_580000.pth"

# # python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id "cuda:2"  --sim_gpu_id "1,2,2,2"  --policy_gpu_id "cuda:1"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000

# # python main.py --auto_gpu_config 0 -n 10  --sem_gpu_id "cuda:3"  --sim_gpu_id "2,2,3,3"  --policy_gpu_id "cuda:1"   --split train  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000
# =======
python main.py --auto_gpu_config 0  -n 8 \
    --sem_gpu_id_list "2"  --policy_gpu_id "cuda:3"  --sim_gpu_id "3" \
    --split train  --backbone_2d "rednet"  --stop_policy "3D" \
    --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml"  --dataset "mp3d" \
    --num_sem_categories 22 --deactivate_entropymap \
    --print_images 1  -d ./tmp  --exp_name exp_kl_goal  --save_periodic 10000 
# >>>>>>> 13a9e4dd5409fa2a5c92cf6db00acbabfe10a7d7
