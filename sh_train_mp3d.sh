
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0  -n 8 \
    --sem_gpu_id_list "0,1"  --policy_gpu_id "cuda:0"  --sim_gpu_id "2" \
    --split train  --backbone_2d "rednet"  --stop_policy "3D" \
    --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml"  --dataset "mp3d" \
    --num_sem_categories 22 --deactivate_entropymap \
    --print_images 1  -d ./tmp  --exp_name exp_kl_goal  --save_periodic 10000 
