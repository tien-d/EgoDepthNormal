##### Run depth demo ####
python main_depth.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
--input_dir ./demo_data/color \
--checkpoint ./checkpoints/edina_depth_baseline.ckpt \
--batch_size 4 --skip_every_n_image_test 1 \
--save_visualization ./demo_visualization/depth_results

##### Run normal demo ####
python main_normal.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
--input_dir ./demo_data/color \
--checkpoint ./checkpoints/edina_normal_baseline.ckpt \
--batch_size 4 --skip_every_n_image_test 1 \
--save_visualization ./demo_visualization/normal_results
