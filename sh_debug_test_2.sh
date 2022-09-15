
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python test_env.py --auto_gpu_config 0 -n 11  --sem_gpu_id "cuda:5"  --sim_gpu_id "4"  --policy_gpu_id "cuda:5"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 0  -d ./tmp --exp_name exp_kl --eval 1  --save_periodic 10000 --dataset "mp3d" --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml"
