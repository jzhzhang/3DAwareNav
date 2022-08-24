
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

<<<<<<< HEAD:sh_debug_test.sh
python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id "cuda:3"  --sim_gpu_id "1,2,2,2"  --policy_gpu_id "cuda:1"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000
=======
python main.py --auto_gpu_config 0 -n 1  --sem_gpu_id_list "0,1"  --sim_gpu_id "1,2,2,2"  --policy_gpu_id "cuda:1"   --split val --backbone_2d  "rednet" --stop_policy "3D"  --print_images 1  -d ./tmp --exp_name exp_kl   --save_periodic 10000
>>>>>>> 80c5486c9008345cd041887fb0c99a1f5f61d549:test_script_server.sh
