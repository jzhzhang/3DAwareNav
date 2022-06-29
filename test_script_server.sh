
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 1  --num_processes_per_gpu 1  --num_processes_on_first_gpu 1 --sim_gpu_id 3 --split train  --print_images 1  -d ./tmp --exp_name exp_test_1 --save_periodic 500000
