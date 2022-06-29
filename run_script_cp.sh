
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 16 --num_processes_per_gpu 8  --num_processes_on_first_gpu 1 --sim_gpu_id 7  -d /DATA/disk1/epic/jiazhaozhang/navigation_data/ObjectGoalNav_data/logs/  --exp_name exp1 --save_periodic 500000
