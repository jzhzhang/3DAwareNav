python main.py --sim_gpu_id 0  --sem_gpu_id "cuda:0" --num_processes 11 --sim_gpu_id "0"  --policy_gpu_id "cuda:0"   --split val --backbone_2d  "rednet" --stop_policy "3D" --split val --eval 1 -d ./tmp --print_images 0 --exp_name eval_3D_stop_policy_rednet  --dataset "mp3d" --num_sem_categories 22 --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml"
--load_2d /home/jiazhaozhang/project/navigation/habitat-challenge/periodic_590000.pth
--load_3d /DATA/disk1/epic/jiazhaozhang/navigation_data/new_tmp/dump/exp_3d_policy_mp-12-09-01:41:22/periodic_3d_420000.pth

