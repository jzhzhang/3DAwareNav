
python main.py  --auto_gpu_config 0  -n 8 \
    --sem_gpu_id_list "2"  --policy_gpu_id "cuda:3"  --sim_gpu_id "3" \
    --backbone_2d "rednet"  --stop_policy "3D"  --deactivate_entropymap \
    --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml"  --dataset "mp3d"  --num_sem_categories 22 \
    --split val  --eval 1  --load ./tmp/dump/exp_base_mp3d-03-09-23:38:48/periodic_1460000.pth \
    --print_images 1  -d ./tmp  --exp_name eval_0901base_mp3d
