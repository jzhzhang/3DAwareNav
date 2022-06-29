
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

python main.py --auto_gpu_config 0 -n 10 --num_processes_per_gpu 10  --num_processes_on_first_gpu 8 --sim_gpu_id 3  -d /DATA/disk1/epic/jiazhaozhang/navigation_data/ObjectGoalNav_data/logs --exp_name exp3 --save_periodic 500000  --load  /DATA/disk1/epic/jiazhaozhang/navigation_data/ObjectGoalNav_data/logs/models/exp2/model_best.pth
