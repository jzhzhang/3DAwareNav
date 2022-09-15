
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

# python main.py --auto_gpu_config 0 -n 8 --sem_gpu_id "cuda:3"  --policy_gpu_id "cuda:0"  --sim_gpu_id "0,0,2,3"  --split train --backbone_2d  "rednet" --stop_policy "3D" --print_images 1  -d ./tmp --exp_name exp_kl_goal  --save_periodic 20000 

python main.py --auto_gpu_config 0 -n 8 --sem_gpu_id_list "0,1"  --policy_gpu_id "cuda:0" \
    --sim_gpu_id "2"  --split train --backbone_2d  "rednet" --stop_policy "3D" \
    --deactivate_entropymap\
    --print_images 1  -d ./new_tmp --exp_name exp_3d_policy_hm  --save_periodic 20000 
    --load_2d /DATA/disk1/epic/jiazhaozhang/navigation_data/new_tmp/dump/exp_3d_policy_mp-09-09-23:22:07/periodic_2d_980000.pth
    --load_3d /DATA/disk1/epic/jiazhaozhang/navigation_data/new_tmp/dump/exp_3d_policy_mp-09-09-23:22:07/periodic_3d_980000.pth
