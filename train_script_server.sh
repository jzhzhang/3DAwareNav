
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

<<<<<<< HEAD
python main.py --auto_gpu_config 0 -n 10 --sem_gpu_id "cuda:3"   --num_processes_per_gpu 5  --num_processes_on_first_gpu 5 --sim_gpu_id "2,2,3,3"  --split train --print_images 1  -d ./tmp --exp_name exp_3D_prob_goal  --load /DATA/disk1/epic/jiazhaozhang/navigation_data/MP_data/models/exp_prob_goal-11-08-20:05:07/model_best.pth  --save_periodic 20000 
=======
python main.py  --auto_gpu_config 0 -n 8 \
    --sem_gpu_id "cuda:2"  --sim_gpu_id 1 \
    --num_processes_per_gpu 8  --num_processes_on_first_gpu 8 \
    --split train  --print_images 0  -d ./tmp \
    --exp_name exp_prob_goal  --save_periodic 300000 \
    --use_recurrent_global 1 \
>>>>>>> 2b3d38e926f0b364113d309de7355e29767e1aff
