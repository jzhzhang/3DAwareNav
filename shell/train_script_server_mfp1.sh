# to the root
cd ..

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py  --auto_gpu_config 0 -n 8\
    --sem_gpu_id "cuda:3"  --policy_gpu_id "cuda:3"  --sim_gpu_id "3" \
    --split train  --print_images 1  -d ./tmp \
    --exp_name exp_rednet_3d  --save_periodic 100000 \
    --backbone_2d  "rednet" --stop_policy "3D" 
