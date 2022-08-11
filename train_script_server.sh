
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 5 --sem_gpu_id "cuda:3"   --num_processes_per_gpu 5  --num_processes_on_first_gpu 5 --sim_gpu_id 1  --split train --print_images 1  -d ./tmp --exp_name exp_prob_goal  --save_periodic 300000 
