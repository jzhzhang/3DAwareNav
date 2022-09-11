
python main.py  --auto_gpu_config 0  -n 8  \
    --sem_gpu_id_list "0"  --policy_gpu_id "cuda:1"  --sim_gpu_id "1" \
    --backbone_2d "rednet"  --stop_policy "3D"  --deactivate_entropymap \
    --split val  --eval 1  --load ./tmp/dump/exp_base_mp3d-03-09-23:38:48/periodic_1460000.pth \
    --print_images 1  -d ./tmp  --exp_name eval_0901base_hm3d
