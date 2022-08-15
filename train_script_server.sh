
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 8 --sem_gpu_id "cuda:3"  --policy_gpu_id "cuda:0"  --sim_gpu_id "1,1,3,3"  --split train --backbone_2d  "rednet" --stop_policy "3D" --print_images 1  -d ./tmp --exp_name exp_kl_goal  --save_periodic 20000 


# python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id "cuda:2"  --sim_gpu_id "1,2,2,2"  --policy_gpu_id "cuda:1"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000

# python main.py --auto_gpu_config 0 -n 10  --sem_gpu_id "cuda:3"  --sim_gpu_id "2,2,3,3"  --policy_gpu_id "cuda:1"   --split train  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000
