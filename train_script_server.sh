
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py  --auto_gpu_config 0 -n 8 \
    --sem_gpu_id "cuda:0"  --sim_gpu_id 1 \
    --num_processes_per_gpu 8  --num_processes_on_first_gpu 8 \
    --split train  --print_images 0  -d ./tmp \
    --exp_name exp_prob_goal  --save_periodic 300000 \
    --use_recurrent_global 1 \
