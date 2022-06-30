#python main_normal.py --train 0 --model_type 'midas_v21' \
#--test_usage 'edina_test' \
#--train_usage 'train' \
#--dataset_pickle_file /mars/home/khiem/projects/ongoing_exps/EgoDepthNormal_Code/dataset_creators/scannet_edina_camready_final_clean.pkl \
#--skip_every_n_image_train 10 --skip_every_n_image_test 40 \
#--batch_size 8 \
#--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/normal_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00011-iter-02000.ckpt \
#--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_eval_release_normal

#python main_depth.py --train 0 --model_type 'midas_v21' \
#--test_usage 'edina_test' \
#--train_usage 'train' \
#--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/depth_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00007-iter-42000.ckpt \
#--dataset_pickle_file /mars/home/khiem/projects/ongoing_exps/EgoDepthNormal_Code/dataset_creators/scannet_edina_camready_final_clean.pkl \
#--batch_size 8 --skip_every_n_image_test 40 \
#--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_eval_release_depth

#python main_depth.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
#--input_dir /mars/mnt/oitstorage/EPIC-KITCHENS/P07/rgb_frames/P07_01 \
#--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/depth_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00007-iter-42000.ckpt \
#--batch_size 4 --skip_every_n_image_test 10 \
#--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_demo_release_depth

#python main_normal.py --train 0 --model_type 'midas_v21' --dataset_type 'demo' \
#--input_dir /mars/mnt/oitstorage/EPIC-KITCHENS/P07/rgb_frames/P07_01 \
#--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/sne_camready_midasv21_baseline/model-epoch-00013-iter-02000.ckpt \
#--batch_size 4 --skip_every_n_image_test 10 \
#--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/midas_randomcrop_epick_p07_01_normal