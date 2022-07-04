#### Run depth evaluation ####
python main_depth.py --train 0 --model_type 'midas_v21' \
--test_usage 'edina_test' \
--checkpoint ./checkpoints/edina_midas_depth_baseline.ckpt \
--dataset_pickle_file ./pickles/scannet_edina_camready_final_clean.pkl \
--batch_size 8 --skip_every_n_image_test 40 \
--data_root PATH/TO/EDINA/DATA \
--save_visualization ./eval_visualization/depth_results

#### Run normal evaluation ####
python main_normal.py --train 0 --model_type 'midas_v21' \
--test_usage 'edina_test' \
--checkpoint ./checkpoints/edina_midas_normal_baseline.ckpt \
--dataset_pickle_file ./pickles/scannet_edina_camready_final_clean.pkl \
--batch_size 8 --skip_every_n_image_test 40 \
--data_root PATH/TO/EDINA/DATA \
--save_visualization ./eval_visualization/normal_results
