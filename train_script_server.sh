
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 10  --num_processes_per_gpu 5  --num_processes_on_first_gpu 5 --sim_gpu_id 7 --split train --print_images 1  -d ./tmp --exp_name exp_test_1 --save_periodic 300000