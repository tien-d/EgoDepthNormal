##### Run depth demo ####
python main_depth.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
--input_dir ./demo_data/color \
--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/depth_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00007-iter-42000.ckpt \
--batch_size 4 --skip_every_n_image_test 1 \
--save_visualization ./demo_visualization/depth_results

##### Run normal demo ####
python main_normal.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
--input_dir ./demo_data/color \
--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/normal_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00011-iter-02000.ckpt \
--batch_size 4 --skip_every_n_image_test 1 \
--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_demodata_normal