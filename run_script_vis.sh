
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 1 --num_processes_per_gpu 1  --num_processes_on_first_gpu 1 --sim_gpu_id 0  -d /data/navigation_data/log/ObjectGoal_test_log  --exp_name exp1 --save_periodic 500000 -v 1
