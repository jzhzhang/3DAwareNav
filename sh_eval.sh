python main.py --sim_gpu_id 0  --sem_gpu_id "cuda:3" --num_processes 8 --sim_gpu_id "0,0,1,1"  --policy_gpu_id "cuda:0"   --split val --backbone_2d  "rednet" --stop_policy "3D" --split val --eval 1 -d ./tmp --load /DATA/disk1/epic/jiazhaozhang/navigation_data/MP_data/dump/exp_kl_goal-17-08-03:11:18/periodic_440000.pth --print_images 1 --exp_name eval_3D_stop_policy_rednet   


# python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id "cuda:2"  --sim_gpu_id "1,2,2,2"  --policy_gpu_id "cuda:1"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000
