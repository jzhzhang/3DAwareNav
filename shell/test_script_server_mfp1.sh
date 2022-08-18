# to the root
cd ..

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 1  \
    --sem_gpu_id "cuda:0"  --policy_gpu_id "cuda:1"  --sim_gpu_id "1" \
    --split train  --print_images 1  -d ./tmp \
    --exp_name exp_test  --save_periodic 100000 \
    --backbone_2d  "rednet" --stop_policy "3D" 