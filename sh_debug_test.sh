
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

# <<<<<<< HEAD
python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id "cuda:5"  --sim_gpu_id "4"  --policy_gpu_id "cuda:5"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 0  -d ./tmp --exp_name exp_kl   --save_periodic 10000
# =======
# python main.py  --auto_gpu_config 0  -n 2 \
#     --sem_gpu_id "cuda:0"  --sim_gpu_id "1"  --policy_gpu_id "cuda:1" \
#     --split val  --backbone_2d "rednet"  --stop_policy "3D" \
#     --print_images 1  -d ./tmp  --exp_name exp_kl  --save_periodic 10000
# >>>>>>> 13a9e4dd5409fa2a5c92cf6db00acbabfe10a7d7

